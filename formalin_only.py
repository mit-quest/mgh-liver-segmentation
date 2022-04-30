import cv2
import math
import numpy as np

SHARPEN_FILTER = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

def sharpen_image(image_path):
    original_image = cv2.imread(image_path)
    sharpened_image = cv2.filter2D(original_image, -1, SHARPEN_FILTER)
    return original_image, sharpened_image

# Assign number between 0 (most likely non-fat) and 1 (most likely fat) for
# probability that island is fat. Input is list of lists of pixels for each island
def get_fat_score_for_watershed_image(islands, original_mask, circularity_threshold=0.5, min_size=10, max_size=1000):
    # Filter islands. Map of tuple of (x, y) coordinates of island to
    # probability that island is fat, initialized to 0.5 (undecided)
    island_to_score_map = dict()
    for island in islands:
        if min_size <= len(island) < max_size: # Filter out small and large tears and holes
            island_to_score_map[tuple(island)] = 0.5
        else:
            island_to_score_map[tuple(island)] = 0 # Small and large tears and holes are assumed to be non-fat (0)

    # --- Assign circularity to islands ---
    island_to_circularity_map = dict() # Map of tuple of (x, y) coordinates of island to circularity score
    for island in island_to_score_map:
        if island_to_score_map[island] > 0:
            # Create mask with 1 for islands and 0 for non-island
            island_mask = np.zeros(original_mask.shape)
            for x, y in island:
                island_mask[x][y] = 1
            island_mask = island_mask.astype(np.uint8)
            contours, hierarchy = cv2.findContours(island_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Find circularity of island
            contour = contours[0]
            contour_area = cv2.contourArea(contour)
            contour_perimeter = cv2.arcLength(contour, True)
            center, radius = cv2.minEnclosingCircle(contour)

            # --- Method to calculate circularity: contour area vs. contour perimeter ---
            circularity = (2 * math.pi * math.sqrt(contour_area / math.pi)) / contour_perimeter

            # Assign circularity
            island_to_circularity_map[island] = circularity

    # --- Filter islands by circularity ---
    for island, circularity in island_to_circularity_map.items():
        if circularity >= circularity_threshold:
            island_to_score_map[island] = 1
        if circularity < circularity_threshold:
            island_to_score_map[island] = 0

    # Create new mask
    new_mask = np.zeros(original_mask.shape)
    for island, circularity in island_to_circularity_map.items():
        if island_to_score_map[island] == 1: # 0 means non-fat, 0.5 means unsure, 1 means fat
            for x, y in island:
                new_mask[x][y] = 255

    new_mask = new_mask.astype(np.uint8)
    return new_mask