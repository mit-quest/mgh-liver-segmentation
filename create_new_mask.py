# This version of get_fat_score is aimed at the "well-defined fat" category

# Assign number between 0 (most likely non-fat) and 1 (most likely fat) for
# probability that island is fat. Input is list of lists of pixels for each island
def get_fat_score(islands, original_mask, min_size=5, max_size=300): # original_mask is np_graph = binary
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

    # Assign islands with high circularity with score of 1, and islands with low circularity with score of 0
    for island, circularity in island_to_circularity_map.items():
        if circularity > 0.8:
            island_to_score_map[island] = 1
        if circularity < 0.4 or (circularity < 0.5 and len(island) > 800): # 20x
            island_to_score_map[island] = 0

    # Assign islands near tears with low or medium circularity with score of 0
    islands_with_score_zero = []
    for island1, island1_score in island_to_score_map.items():
        # Find islands classified as non-fat
        if island1_score == 0 and len(island1) > min_size:
            islands_with_score_zero.append(island1)

    for island1 in islands_with_score_zero:
        for island2, island2_circularity in island_to_circularity_map.items():
            island1_island2_distance = np.amin(distance.cdist(np.array(island1), np.array(island2)).min(axis=1))
            if island1 != island2:
                # Islands near large tears are more likely to be tears
                if max_size <= len(island1) < 2000 and island1_island2_distance < 50 and island2_circularity <= 0.9:
                    island_to_score_map[island2] = 0
                if island1_island2_distance < 10 and island2_circularity < 0.6: # 20x
                    island_to_score_map[island2] = 0

    # Assign islands near fat with medium or high circularity with score of 1
    islands_with_score_one = []
    for island1, island1_score in island_to_score_map.items():
        if island1_score == 1: # Find islands classified as non-fat
            islands_with_score_one.append(island1)

    for island1 in islands_with_score_one:
        for island2, island2_circularity in island_to_circularity_map.items():
            island1_island2_distance = np.amin(distance.cdist(np.array(island1), np.array(island2)).min(axis=1))
            island1_island2_ratio = len(island1) / len(island2)
            if island1 != island2 and island1_island2_distance < 20 and island_to_score_map[island2] == 0.5 and island1_island2_ratio > 0.3:
                island_to_score_map[island2] = 1

    # Filter unassigned islands
    for island, circularity in island_to_circularity_map.items():
        if island_to_score_map[island] == 0.5 and 100 < len(island) < 800 and circularity > 0.5:
            island_to_score_map[island] = 1

    # Create new mask
    new_mask = np.zeros(original_mask.shape)
    for island, circularity in island_to_circularity_map.items():
        if island_to_score_map[island] == 1: # 0 means non-fat, 0.5 means unsure, 1 means fat
            for x, y in island:
                new_mask[x][y] = 255

        # Code for visualizing identified islands when improving masks
        # if island_to_score_map[island] != 1 and 100 < len(island) < 800 and circularity > 0.5:
        #     for x, y in island:
        #         new_mask[x][y] = 255
    new_mask = new_mask.astype(np.uint8)
    return new_mask
