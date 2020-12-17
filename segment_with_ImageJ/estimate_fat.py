def count_fat_macro_micro(image, mask):
    image_255 = copy.deepcopy(image)*255 # Use cv2_imshow to show image
    image_inv = copy.deepcopy(image)

    # Find area of black background in original image
    num_black_border_pixels = 0
    for row in range(len(image_255)):
        for col in range(len(image_255[row])):
            if image_255[row][col] <= 45 and not (170 <= row <= 520 and 70 <= col <= 410): # 45 experimentally determined
                image_inv[row][col] = 255
            else:
                image_inv[row][col] = 0
    _, binary_image_inv = cv2.threshold(image_inv, 200, 255, cv2.THRESH_BINARY) # 200 was chosen to separate 0 (black) and 255 (white) pixels
    binary_image_inv = binary_opening(binary_image_inv, disk(1))
    image_np_graph_inv = binary_image_inv
    image_row_inv, image_col_inv = image_np_graph_inv.shape
    image_graph_inv = image_np_graph_inv.tolist()
    image_g_inv = Graph(image_row_inv, image_col_inv, image_graph_inv)
    image_islands_inv = image_g_inv.findIslands()
    for island in image_islands_inv:
        if len(island) > 1000: # 1000 was chosen as an arbitrarily large number of pixels
            num_black_border_pixels += len(island)

    # Find area of large white islands
    num_white_island_pixels = 0
    _, binary_image = cv2.threshold(image_255, 200, 255, cv2.THRESH_BINARY) # 150 experimentally determined
    binary_image = binary_opening(binary_image, disk(1))
    image_np_graph = binary_image
    image_row, image_col = image_np_graph.shape
    image_graph = image_np_graph.tolist()
    image_g = Graph(image_row, image_col, image_graph)
    image_islands = image_g.findIslands()
    for island in image_islands:
        if len(island) > 700: # 700 was experimentally determined
            num_white_island_pixels += len(island)

    # Find liver area excluding border and large tears
    liver_area = (image_255.shape[0] * image_255.shape[1]) - num_black_border_pixels - num_white_island_pixels

    # Find fat area using predicted mask
    fat_areas = []
    mask_255 = copy.deepcopy(mask)*255
    mask_macro = copy.deepcopy(mask) # Not needed but helpful for debugging
    _, binary_mask = cv2.threshold(mask_255, 30, 255, cv2.THRESH_BINARY) # 30 experimentally determined
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

    # Calculate fat estimates with liver area not including fat area
    # total_fat_percentage = total_fat_area / (liver_area-total_fat_area) * 100
    # macro_fat_percentage = total_macro_area / (liver_area-total_macro_area) * 100

    return total_fat_percentage, macro_fat_percentage


def mean_fat_percent(image_names, original_folder, new_mask_folder):
    # Get original images
    image_list = []
    for liver_file in image_names:
        image_file = f"{original_folder}/{liver_file}"
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
        mask_file = f"{new_mask_folder}/{liver_file}"
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
        total_fat, macro_fat = count_fat_macro_micro(images[i], masks[i])
        liver_total_fat.append(total_fat)
        liver_macro_fat.append(macro_fat)

    # Calculate min, max, mean, and median total fat
    min_total_fat = min(liver_total_fat)
    max_total_fat = max(liver_total_fat)
    mean_total_fat = sum(liver_total_fat) / num_liver_files
    median_total_fat = median(liver_total_fat)
    print("TOTAL FAT: ", " Mean: ", round(mean_total_fat, 2), " Median: ", round(median_total_fat, 2))

    # Calculate min, max, mean, and median macro fat
    min_macro_fat = min(liver_macro_fat)
    max_macro_fat = max(liver_macro_fat)
    mean_macro_fat = sum(liver_macro_fat) / num_liver_files
    median_macro_fat = median(liver_macro_fat)
    print("MACRO FAT: ", "Mean: ", round(mean_macro_fat, 2), " Median: ", round(median_macro_fat, 2))

    return round(mean_total_fat, 2), round(mean_macro_fat, 2)

image_names = ['HF-8 formalin - 3h - 20x - d - 1.tiff']
mean_fat_percent(image_names,
                 'drive/MyDrive/ImageJ-original-images',
                 'drive/MyDrive/ImageJ-new-mask-images')
