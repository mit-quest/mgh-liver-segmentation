# Assign number between 0 (most likely non-fat) and 1 (most likely fat) for
# probability that island is fat. Input is list of lists of pixels for each island
def get_fat_score_for_watershed_image(islands, original_mask, min_size=10, max_size=1000):
    # Filter islands. Map of tuple of (x, y) coordinates of island to
    # probability that island is fat, initialized to 0.5 (undecided)
    island_to_score_map = dict()
    for island in islands:
        if min_size <= len(island) < max_size: # Filter out small and large tears and holes
            island_to_score_map[tuple(island)] = 0.5
        else:
            island_to_score_map[tuple(island)] = 0 # Small and large tears and holes are assumed to be non-fat (0)

    island_to_circularity_map = dict() # Map of tuple of (x, y) coordinates of island to circularity score
    for island in island_to_score_map:
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
        circularity = (2 * math.pi * math.sqrt(contour_area / math.pi)) / contour_perimeter
        island_to_circularity_map[island] = circularity

    # Thresholds for general liver images
    # for island, circularity in island_to_circularity_map.items():
    #     if circularity > 0.85:                       # Threshold determined experimentally. Set to >0.85
    #         island_to_score_map[island] = 1
    #     if circularity < 0.65 and len(island) < 260: # Threshold determined experimentally. Set to <0.65 and <260
    #         island_to_score_map[island] = 0

    # Thresholds for HF-10 formalin - 6h - 20x - f - 1.tiff
    # for island, circularity in island_to_circularity_map.items():
    #     if circularity > 0.8:
    #         island_to_score_map[island] = 1
    #     if circularity < 0.65:
    #         if len(island) < 260:
    #             island_to_score_map[island] = 0
    #         else:
    #             island_to_score_map[island] = 1
    #     if island_to_score_map[island] == 0.5 and len(island) < 20:
    #         island_to_score_map[island] = 0
    #     if island_to_score_map[island] == 0.5:
    #         if circularity < 0.7:
    #             if len(island) < 185:
    #                 island_to_score_map[island] = 0
    #             else:
    #                 island_to_score_map[island] = 1
    #         else:
    #             island_to_score_map[island] = 1

    # Thresholds for HF-21 formalin - 3h - 20x - g - 1.tiff
    # for island, circularity in island_to_circularity_map.items():
    #     if circularity > 0.8:
    #         island_to_score_map[island] = 1
    #     if (circularity < 0.7 and len(island) < 60) or (circularity < 0.6 and len(island) < 260):
    #         island_to_score_map[island] = 0
    #     if (island_to_score_map[island] == 0.5) and (len(island) < 20):
    #         island_to_score_map[island] = 0
    #     if (island_to_score_map[island] == 0.5) and (len(island) > 100 or circularity > 0.75):
    #         island_to_score_map[island] = 1

    # Thresholds for HF-3 formalin - pre - 20x - a - 1.tiff
    # for island, circularity in island_to_circularity_map.items():
    #     if circularity > 0.8:
    #         island_to_score_map[island] = 1
    #     if circularity < 0.65:
    #         island_to_score_map[island] = 0

    # Thresholds for HF-3 formalin - pre - 20x - c - 1.tiff and HF-3 formalin - 12h - 20x - e - 1.tiff
    for island, circularity in island_to_circularity_map.items():
        if circularity > 0.75:
            island_to_score_map[island] = 1
        if circularity < 0.65:
            island_to_score_map[island] = 0

    # Create new mask
    new_mask = np.zeros(original_mask.shape)
    for island, circularity in island_to_circularity_map.items():
        if island_to_score_map[island] == 1: # 0 means non-fat, 0.5 means unsure, 1 means fat
            # print((len(island), circularity))
            for x, y in island:
                new_mask[x][y] = 255
    new_mask = new_mask.astype(np.uint8)
    return new_mask
