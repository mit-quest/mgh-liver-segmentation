import copy
import csv
import cv2
import math
import numpy as np
import os
from PIL import Image
from skimage.morphology import disk, binary_opening
from skimage.transform import rescale, resize
from statistics import median

import common


def _calculate_liver_area(image_path, mag_vars):
    """
    Calculate the number of pixels corresponding to liver tissue
    """
    src = cv2.imread(image_path)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (mag_vars['blur_val'], mag_vars['blur_val']))

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

def _calculate_large_white_area(image_path, is_frozen, mag_vars):
    """
    Calculate the number of pixels corresponding to large white areas (tears or not liver tissue)
    """
    image = cv2.imread(image_path)

    erode_image_bool = common.prepare_image(image_path, is_frozen, mag_vars)

    islands = common.new_graph(erode_image_bool)
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

            if (island_len > mag_vars['island_len_upper_frozen']) or (island_len > mag_vars['island_len_lower_frozen'] and ratio < 0.3):
                num_large_white_area_pixels += island_len
                for x, y in island:
                    large_white_area_array[x][y] = 255

    return num_large_white_area_pixels, large_white_area_array

def _count_fat_macro_micro(image, mask, image_path, is_frozen, mag_vars):
    image_255 = copy.deepcopy(image)*255 # Use cv2_imshow to show image
    image_inv = copy.deepcopy(image)

    # Find area of black background in original image
    num_black_border_pixels, _ = _calculate_liver_area(image_path, mag_vars)

    # Find area of large white islands
    num_white_island_pixels, _ = _calculate_large_white_area(image_path, is_frozen, mag_vars)

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
    mask_g = common.Graph(mask_row, mask_col, mask_graph)
    mask_islands = mask_g.findIslands()
    for island in mask_islands:
        fat_areas.append(len(island))
        if len(island) >= mag_vars['macro_fat_lower']:
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
        if fat_area >= mag_vars['macro_fat_lower']:
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

def _mean_fat_percent(image_names, original_folder, new_mask_folder, is_frozen, mag_vars):
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
        total_fat, macro_fat = _count_fat_macro_micro(images[i], masks[i], image_paths[i], is_frozen, mag_vars)
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

def estimate_steatosis(images_directory, output_directory, liver_name, image_name, is_frozen, mag_vars):	
    original_image_path = os.path.join(images_directory, liver_name)
    mask_image_path = os.path.join(output_directory, liver_name)
    image_path = os.path.join(images_directory, liver_name, image_name)
    mean_total_fat, mean_macro_fat = _mean_fat_percent([image_name], original_image_path, mask_image_path, is_frozen, mag_vars)

    # Save fat estimates per liver to CSV file. Save in order of image name, total fat, macro fat (arbitrary threshold)
    csv_save_path = os.path.join(output_directory, liver_name, f'{liver_name}_fat_estimates.csv')
    with open(csv_save_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([image_name, mean_total_fat, mean_macro_fat])

    _, large_white_area_array = _calculate_large_white_area(image_path, is_frozen, mag_vars)
    _, black_background_area_array = _calculate_liver_area(image_path, mag_vars)
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
