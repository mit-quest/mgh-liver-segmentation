import cv2
from datetime import timedelta
import numpy as np
import math
import argparse
import time
from typing import List, Dict, Any
from statistics import median

from skimage import io, data, color, img_as_ubyte
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.draw import ellipse_perimeter, line
from skimage.measure import find_contours
from skimage.morphology import remove_small_holes, remove_small_objects, binary_erosion, binary_dilation
from skimage.util import img_as_ubyte
from skimage import color, img_as_float
import queue

from scipy import ndimage as ndi
from scipy.spatial import distance

import os
import sys
sys.setrecursionlimit(10000000)

import imagej

import common
import frozen_only
import formalin_only


def _run_watershed(binary_image_path, watershed_image_path, ij):
    macro = """
    //@ String binary_image_path
    //@ String watershed_image_path

    open(binary_image_path);
    run("8-bit");
    run("Watershed");
    saveAs("Tiff", watershed_image_path);
    """
    args = {
        'binary_image_path': binary_image_path,
        'watershed_image_path': watershed_image_path + '-w2.tiff',
    };
    result = ij.py.run_macro(macro, args);
    result.getOutput('watershed_image_path')

def _watershed_and_mask(save_path, ij):

    # Run watershed and save image
    _run_watershed(save_path, save_path, ij)

    # After applying watershed
    watershed_binary = cv2.imread(save_path)
    watershed_gray = cv2.cvtColor(watershed_binary, cv2.COLOR_BGR2GRAY)
    watershed_bool = watershed_gray > 0

    # Create mask
    islands = common.new_graph(watershed_bool) # Use eroded watershed binary

    return islands, watershed_bool

def _remove_background(opened_image_bool, is_frozen):
    islands = common.new_graph(opened_image_bool)

    # Filter islands by size. Keep islands with sizes less than max_size
    island_to_score_map = dict()
    for island in islands:
        island_len = len(island)
        if island_len > 0:
            # Create mask with 1 for islands and 0 for non-island
            island_mask = np.zeros(opened_image_bool.shape)
            for x, y in island:
                island_mask[x][y] = 1
            island_mask = island_mask.astype(np.uint8)
            contours, hierarchy = cv2.findContours(island_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Find circularity of island
            contour = contours[0]
            contour_area = cv2.contourArea(contour)
            contour_perimeter = cv2.arcLength(contour, True)
            center, radius = cv2.minEnclosingCircle(contour)
            ratio = contour_area / (math.pi * radius * radius)

            if is_frozen:
                if (island_len > 2000) or (island_len > 600 and ratio < 0.3):
                    island_to_score_map[tuple(island)] = 0
                else:
                    island_to_score_map[tuple(island)] = 1
            else:
                if island_len < 1000 and ratio > 0.17: # 1000 good for formalin HF-3 liver, 0.17 determined experimentally
                    island_to_score_map[tuple(island)] = 1 # Score of 1 means keep
                else:
                    island_to_score_map[tuple(island)] = 0 # Score of 0 means do not keep

    # Create binary image without obvious non-fat. Color black non-fat islands
    binary_without_obvious_non_fat = np.zeros(opened_image_bool.shape)
    for island, score in island_to_score_map.items():
        if score == 1:
            for x, y in island:
                binary_without_obvious_non_fat[x][y] = 255
    binary_without_obvious_non_fat = binary_without_obvious_non_fat.astype(np.uint8)
    return binary_without_obvious_non_fat

def segment_liver(images_directory, output_directory, liver_name, image_name, is_frozen, ij):
    start_time = time.monotonic()

    image_path = os.path.join(images_directory, liver_name, image_name)
    erode_image_bool = common.prepare_image(image_path, is_frozen)

    # Find binary image without background, save binary
    print("Isolating background...")
    binary_without_obvious_non_fat = _remove_background(erode_image_bool, is_frozen)
    save_path = os.path.join(output_directory, liver_name, image_name)
    cv2.imwrite(save_path + '-binary.tiff', binary_without_obvious_non_fat)
    cv2.imwrite(save_path, binary_without_obvious_non_fat)
    print("Completed in " + str(timedelta(seconds=time.monotonic() - start_time)))

    # run watershed and create mask
    print("Running watershed...")
    islands, np_graph = _watershed_and_mask(save_path, ij)

    # Create new mask, convert to overlay with green mask
    new_mask = common.get_fat_score_for_watershed_image(islands, np_graph, circularity_threshold=0.6, min_size=2, max_size=1000) # With erosion, adjust size thresholds
    new_mask_dilate_bool = binary_dilation(new_mask)
    print("Completed in " + str(timedelta(seconds=time.monotonic() - start_time)))
    
    if is_frozen:
        new_mask_dilate = new_mask_dilate_bool.astype(np.uint8)
        new_mask_dilate *= 255

        # extra watershed and circularity step
        cv2.imwrite(save_path + '-watershed.tiff', new_mask_dilate)
        cv2.imwrite(save_path, new_mask_dilate)

        # run watershed and create mask again
        print("Running extra watershed...")
        islands_new, np_graph_new = _watershed_and_mask(save_path, ij)
        
        # Create new mask, convert to overlay with green mask
        new_mask = common.get_fat_score_for_watershed_image(islands_new, np_graph_new, circularity_threshold=0.7, contour_area_vs_perimeter=True, min_size=2, max_size=500) # With erosion, adjust size thresholds
        print("Completed in " + str(timedelta(seconds=time.monotonic() - start_time)))
    else:
        new_mask = new_mask_dilate_bool

    new_mask_float = img_as_float(new_mask)
    green_multiplier = [0, 1, 0]
    new_mask_rgb = color.gray2rgb(new_mask_float) * green_multiplier

    # Save mask
    new_mask_to_save = new_mask_rgb.astype(np.uint8) * 255
    cv2.imwrite(save_path + '-new_mask.tiff', new_mask_to_save)
    cv2.imwrite(save_path, new_mask_to_save)
