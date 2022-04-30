import cv2
import numpy as np
import math
import csv
import argparse
from typing import List, Dict, Any
from statistics import median

from skimage import io, data, color, img_as_ubyte
from skimage.color import rgb2gray
from skimage.morphology import binary_opening, disk, remove_small_holes, remove_small_objects, binary_erosion, binary_dilation
from skimage.transform import rescale, resize
from skimage.feature import canny
from skimage.draw import ellipse_perimeter, line
from skimage.measure import find_contours
from skimage.util import img_as_ubyte
from skimage import color, img_as_float
import queue
import copy

from scipy import ndimage as ndi
from scipy.spatial import distance
from PIL import Image

import os
import sys
sys.setrecursionlimit(10000000)

import imagej

import frozen_only
import formalin_only

class Graph:
    def __init__(self, row, col, graph):
        self.ROW = row
        self.COL = col
        self.graph = graph

    def isSafe(self, i, j, visited):
        # Row number is in range, column number is in range, value is 1 and not yet visited
        return 0 <= i and i < self.ROW and 0 <= j and j < self.COL and not visited[i][j] and self.graph[i][j]

    def BFS(self, i, j, visited, island):
        # Utility function to do BFS for a 2D boolean matrix. Uses only the 4 neighbors as adjacent vertices
        rowNbr = [-1, 0, 1, 0]
        colNbr = [0, -1, 0, 1]
        q = []
        q.append((i,j))
        visited[i][j] = True

        while len(q) != 0:
            x,y = q.pop(0)
            for k in range(len(rowNbr)):
                if self.isSafe(x + rowNbr[k], y + colNbr[k], visited):
                    island.append((x + rowNbr[k], y + colNbr[k]))
                    visited[(x) + rowNbr[k]][y + colNbr[k]] = True
                    q.append((x + rowNbr[k], y + colNbr[k]))

    def findIslands(self):
        # Make a bool array to mark visited cells. Initially all cells are unvisited
        visited = [[False for j in range(self.COL)]for i in range(self.ROW)]
        # Initialize count as 0 and traverse through cells of given matrix
        index = 0
        islands = []
        for i in range(self.ROW):
            for j in range(self.COL):
                # If a cell with value 1 is not visited yet, then new island found
                if visited[i][j] == False and self.graph[i][j] == 1:
                    # Visit all cells in this island and increment island count
                    island = []
                    self.BFS(i, j, visited, island)
                    islands.append(island)
                    index += 1
        return islands

def _prepare_image(image_path, is_frozen):
    if is_frozen:
        white_areas_in_liver_tissue = frozen_only.find_white_areas_in_liver_tissue(image_path)
        bool_input = white_areas_in_liver_tissue
    else:
        # Find sharpened, grayscale, and binary images
        original_image, sharpened_image = sharpen_image(image_path)
        sharpened_gray = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
        _, sharpened_binary = cv2.threshold(sharpened_gray, 195, 255, cv2.THRESH_BINARY)
        bool_input = sharpened_binary

    # Remove noise
    opened_image_bool = remove_small_holes(bool_input, area_threshold=2) # Change black area < 5 pixels to white
    opened_image_bool = remove_small_objects(opened_image_bool, min_size=10) # Change white area < 10 pixels to black
    opened_image = opened_image_bool.astype(np.uint8)  # Convert to an unsigned byte
    opened_image *= 255

    # Apply erosion
    erode_image_bool = binary_erosion(opened_image_bool)
    erode_image = erode_image_bool.astype(np.uint8)
    erode_image *= 255

    return erode_image_bool

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
    np_graph = watershed_bool # Use eroded watershed binary
    row, col = np_graph.shape
    graph = np_graph.tolist()
    g = Graph(row, col, graph)
    islands = g.findIslands()

    return islands, np_graph

def _remove_background(opened_image_bool, is_frozen):
    np_graph = opened_image_bool
    row, col = np_graph.shape
    graph = np_graph.tolist()
    g = Graph(row, col, graph)
    islands = g.findIslands()

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

def _calculate_large_white_area(image_path, is_frozen):
    """
    Calculate the number of pixels corresponding to large white areas (tears or not liver tissue)
    """
    image = cv2.imread(image_path)
    erode_image_bool = _prepare_image(image_path, is_frozen)

    np_graph = erode_image_bool
    row, col = np_graph.shape
    graph = np_graph.tolist()
    g = Graph(row, col, graph)
    islands = g.findIslands()

    num_large_white_area_pixels = 0
    large_white_area_array = np.zeros((image.shape[0], image.shape[1]))
    for island in islands:
        island_len = len(island)
        if island_len > 0:
            island_mask = np.zeros(erode_image_bool.shape)
            for x, y in island:
                island_mask[x][y] = 1
            island_mask = island_mask.astype(np.uint8)
            contours, hierarchy = cv2.findContours(island_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            contour = contours[0]
            contour_area = cv2.contourArea(contour)
            contour_perimeter = cv2.arcLength(contour, True)
            center, radius = cv2.minEnclosingCircle(contour)
            ratio = contour_area / (math.pi * radius * radius)

            if (island_len > 2000) or (island_len > 600 and ratio < 0.3):
                num_large_white_area_pixels += island_len
                for x, y in island:
                    large_white_area_array[x][y] = 255
    return num_large_white_area_pixels, large_white_area_array

def _calculate_liver_area(image_path):
    """
    Calculate the number of pixels corresponding to liver tissue
    """
    src = cv2.imread(image_path)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (3, 3))

    # Apply Hough transform
    detected_circles = cv2.HoughCircles(gray_blurred,
                       cv2.HOUGH_GRADIENT, 1, 400, param1 = 50,
                       param2 = 30, minRadius = 230, maxRadius = 300)

    # Draw detected circles
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            cv2.circle(src, (a, b), r, (0, 255, 0), 2)

        num_background_pixels = 0
        num_rows, num_cols, _ = src.shape
        background_area_array = np.zeros((num_rows, num_cols))
        for i in range(num_rows):
            for j in range(num_cols):
                if not (((i - b)**2 + (j - a)**2) < r**2): # not inside circle
                    num_background_pixels += 1
                    background_area_array[i][j] = 255
    return num_background_pixels, background_area_array

def _count_fat_macro_micro(image, mask, image_path, is_frozen):
    image_255 = copy.deepcopy(image)*255 # Use cv2_imshow to show image
    image_inv = copy.deepcopy(image)

    # Find area of black background in original image
    num_black_border_pixels, _ = _calculate_liver_area(image_path)

    # Find area of large white islands
    num_white_island_pixels, _ = _calculate_large_white_area(image_path, is_frozen)

    # Find liver area excluding border and large tears
    liver_area = (image_255.shape[0] * image_255.shape[1]) - num_black_border_pixels - num_white_island_pixels

    # Find fat area using predicted mask
    fat_areas = []
    mask_255 = copy.deepcopy(mask)*255
    mask_macro = copy.deepcopy(mask) # Not needed but helpful for debugging
    _, binary_mask = cv2.threshold(mask_255, 30, 255, cv2.THRESH_BINARY)
    binary_mask = binary_opening(binary_mask, disk(1))
    mask_np_graph = binary_mask
    mask_row, mask_col = mask_np_graph.shape
    mask_graph = mask_np_graph.tolist()
    mask_g = Graph(mask_row, mask_col, mask_graph)
    mask_islands = mask_g.findIslands()
    for island in mask_islands:
        fat_areas.append(len(island))
        if len(island) >= 30:
            for row, col in island:
                mask_macro[row][col] = 255
        else:
            for row, col in island:
                mask_macro[row][col] = 150
    num_fat = len(fat_areas)
    total_fat_area = sum(fat_areas)

    macro = []
    micro = []
    for fat_area in fat_areas:
        if fat_area >= 30:
            macro.append(fat_area)
        else:
            micro.append(fat_area)
    num_macro = len(macro)
    num_micro = len(micro)
    total_macro_area = sum(macro)
    total_micro_area = sum(micro)

    # Calculate fat estimates with total liver area
    total_fat_percentage = total_fat_area / liver_area * 100
    macro_fat_percentage = total_macro_area / liver_area * 100
    return total_fat_percentage, macro_fat_percentage

def _mean_fat_percent(image_names, original_folder, new_mask_folder, is_frozen):
    # Get original images
    image_list = []
    image_paths = []
    for liver_file in image_names:
        image_file = os.path.join(original_folder, liver_file)
        image_paths.append(image_file)
        original_image = Image.open(image_file).convert('L')
        if original_image.size != (480, 640):
            image_list.append(np.array(original_image.resize((480,640))))
        else:
            image_list.append(np.array(original_image))
    image_np = np.asarray(image_list)
    images = np.asarray(image_np, dtype=np.float32)/255
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)

    # Get new masks
    mask_list = []
    for liver_file in image_names:
        mask_file = os.path.join(new_mask_folder, f'{liver_file}')
        new_mask = Image.open(mask_file).convert('L')
        if new_mask.size != (480, 640):
            mask_list.append(np.array(new_mask.resize((480,640))))
        else:
            mask_list.append(np.array(new_mask))
    mask_np = np.asarray(mask_list)
    masks = np.asarray(mask_np, dtype=np.float32)/255
    masks = masks.reshape(masks.shape[0], masks.shape[1], masks.shape[2], 1)

    # Find mean total and macro fat percent
    liver_total_fat = []
    liver_macro_fat = []
    num_liver_files = len(image_names)
    for i in range(num_liver_files):
        total_fat, macro_fat = _count_fat_macro_micro(images[i], masks[i], image_paths[i], is_frozen)
        liver_total_fat.append(total_fat)
        liver_macro_fat.append(macro_fat)

    # Calculate min, max, mean, and median total fat
    min_total_fat = min(liver_total_fat)
    max_total_fat = max(liver_total_fat)
    mean_total_fat = sum(liver_total_fat) / num_liver_files
    median_total_fat = median(liver_total_fat)

    # Calculate min, max, mean, and median macro fat
    min_macro_fat = min(liver_macro_fat)
    max_macro_fat = max(liver_macro_fat)
    mean_macro_fat = sum(liver_macro_fat) / num_liver_files
    median_macro_fat = median(liver_macro_fat)

    return round(mean_total_fat, 2), round(mean_macro_fat, 2)

def segment_liver(images_directory, output_directory, liver_name, image_name, preservation, ij):
    is_frozen = True if preservation == 'frozen' else False

    print("Starting " + image_name)
    image_path = os.path.join(images_directory, liver_name, image_name)

    erode_image_bool = _prepare_image(image_path, is_frozen)

    # Find binary image without background, save binary
    print("Isolating background...")
    binary_without_obvious_non_fat = _remove_background(erode_image_bool, is_frozen)
    save_path = os.path.join(output_directory, liver_name, image_name)
    cv2.imwrite(save_path + '-binary.tiff', binary_without_obvious_non_fat)
    cv2.imwrite(save_path, binary_without_obvious_non_fat)

    # run watershed and create mask
    print("Running watershed...")
    islands, np_graph = _watershed_and_mask(save_path, ij)

    # Create new mask, convert to overlay with green mask
    if is_frozen:
        new_mask = frozen_only.get_fat_score_for_watershed_image(islands, np_graph, circularity_threshold=0.6, min_size=2, max_size=1000) # With erosion, adjust size thresholds
    else:
        new_mask = formalin_only.get_fat_score_for_watershed_image(islands, np_graph, circularity_threshold=0.6, min_size=2, max_size=1000) # With erosion, adjust size thresholds
    new_mask_dilate_bool = binary_dilation(new_mask)
    
    if is_frozen:
        new_mask_dilate = new_mask_dilate_bool.astype(np.uint8)
        new_mask_dilate *= 255

        # New: Extra watershed and circularity step
        cv2.imwrite(save_path + '-watershed.tiff', new_mask_dilate)
        cv2.imwrite(save_path, new_mask_dilate)

        # run watershed and create mask again
        print("Running extra watershed...")
        islands_new, np_graph_new = _watershed_and_mask(save_path, ij)
        
        # Create new mask, convert to overlay with green mask
        new_mask = frozen_only.get_fat_score_for_watershed_image(islands_new, np_graph_new, circularity_threshold=0.7, contour_area_vs_perimeter=True, min_size=2, max_size=500) # With erosion, adjust size thresholds
    else:
        new_mask = new_mask_dilate_bool

    new_mask_float = img_as_float(new_mask)
    green_multiplier = [0, 1, 0]
    new_mask_rgb = color.gray2rgb(new_mask_float) * green_multiplier

    # Save mask
    new_mask_to_save = new_mask_rgb.astype(np.uint8) * 255
    cv2.imwrite(save_path + '-new_mask.tiff', new_mask_to_save)
    cv2.imwrite(save_path, new_mask_to_save)

    # Estimate fat
    print("Estimating steatosis...")
    original_image_path = os.path.join(images_directory, liver_name)
    mask_image_path = os.path.join(output_directory, liver_name)
    mean_total_fat, mean_macro_fat = _mean_fat_percent([image_name], original_image_path, mask_image_path, is_frozen)

    # Save fat estimates per liver to CSV file. Save in order of image name, total fat, macro fat (arbitrary threshold)
    csv_save_path = os.path.join(output_directory, liver_name, f'{liver_name}_fat_estimates.csv')
    with open(csv_save_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([image_name, mean_total_fat, mean_macro_fat])

    _, large_white_area_array = _calculate_large_white_area(image_path, is_frozen)
    _, black_background_area_array = _calculate_liver_area(image_path)
    combined_non_liver_array = np.maximum(large_white_area_array, black_background_area_array)
    combined_non_liver_array = 255 - combined_non_liver_array
    combined_non_liver_image = Image.fromarray(np.uint8(combined_non_liver_array) , 'L')
    combined_non_liver_image = combined_non_liver_image.convert("RGBA")

    background_path = os.path.join(original_image_path, image_name)
    background = Image.open(background_path)
    overlay_path = os.path.join(output_directory, liver_name, image_name)
    overlay = Image.open(overlay_path)
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    new_img = Image.blend(background, combined_non_liver_image, 0.5) # Add large white area image on top of original image
    new_img = Image.blend(new_img, overlay, 0.5) # Add segmented fat mask on top of current image
    new_img.save(overlay_path, "tiff")
    print(image_name + " complete!")
