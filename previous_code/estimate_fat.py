from statistics import median


def count_fat_macro_micro(image, mask):
    image_255 = image*255
#     plt.imshow(image_255, cmap='Greys_r')

    # Find area of black background in original image
    num_black_border_pixels = 0
    for row in range(len(image_255)):
        for col in range(len(image_255[row])):
            if image_255[row][col] <= 35 and not (170 <= row <= 520 and 70 <= col <= 410): # 30 was experimentally determined to remove border area
                num_black_border_pixels += 1

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
    liver_area = (image.shape[0] * image.shape[1]) - num_black_border_pixels - num_white_island_pixels

    # Find fat area using predicted mask
    fat_areas = []
    mask_255 = mask*255
    _, binary_mask = cv2.threshold(mask_255, 30, 255, cv2.THRESH_BINARY) # 30 experimentally determined
    binary_mask = binary_opening(binary_mask, disk(1))

    mask_np_graph = binary_mask
    mask_row, mask_col = mask_np_graph.shape
    mask_graph = mask_np_graph.tolist()
    mask_g = Graph(mask_row, mask_col, mask_graph)
    mask_islands = mask_g.findIslands()
    for island in mask_islands:
        fat_areas.append(len(island))
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


def get_image_files_from_folders(liver_folders, include_path=False):
    liver_files = []
    for liver_folder in liver_folders:
        for blob in bucket.list_blobs(prefix=f"raw-data/{liver_folder}"):
            blob_name_split = blob.name.split("/")
            file = blob_name_split[2]
            if blob_name_split[2] != "" and "20x" in file and f"{liver_folder} " in file: # Include only 20x images
                if include_path and blob.name not in liver_files:
                    liver_files.append(blob.name)
                else:
                    if blob_name_split[2] not in liver_files:
                        liver_files.append(blob_name_split[2])
    return liver_files


def mean_fat_percent(liver):
    # liver_files is a list of file names for original images
    # liver_files = get_image_files_from_folders([liver]) # Old code before moving to Colab
    liver_files = ['HF-21 formalin - 3h - 20x - g - 1.tiff']

    # Get original images
    image_list = []
    for liver_file in liver_files:
        # image_file is the file path to the original image
        # image_file = "bucket/raw-data/" + liver + "/" + liver_file
        image_file = "drive/MyDrive/" + liver_file
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
    for liver_file in liver_files:
        # mask_file = "bucket/new-masks-20x/" + liver_file
        mask_file = "drive/MyDrive/new-masks-20x_HF-21 formalin - 3h - 20x - g - 1 copy.tiff"
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
    num_liver_files = len(liver_files)
    for i in range(num_liver_files):
        total_fat, macro_fat = count_fat_macro_micro(images[i], masks[i])
        liver_total_fat.append(total_fat)
        liver_macro_fat.append(macro_fat)

    # Calculate min, max, mean, and median total fat
    min_total_fat = min(liver_total_fat)
    max_total_fat = max(liver_total_fat)
    mean_total_fat = sum(liver_total_fat) / num_liver_files
    median_total_fat = median(liver_total_fat)
    print("TOTAL FAT Min: ", round(min_total_fat, 2), " Max: ", round(max_total_fat, 2), " Mean: ", round(mean_total_fat, 2), " Median: ", round(median_total_fat, 2))

    # Calculate min, max, mean, and median macro fat
    min_macro_fat = min(liver_macro_fat)
    max_macro_fat = max(liver_macro_fat)
    mean_macro_fat = sum(liver_macro_fat) / num_liver_files
    median_macro_fat = median(liver_macro_fat)
    print("MACRO FAT Min: ", round(min_macro_fat, 2), " Max: ", round(max_macro_fat, 2), " Mean: ", round(mean_macro_fat, 2), " Median: ", round(median_macro_fat, 2))

    return round(mean_total_fat, 2), round(mean_macro_fat, 2)

mean_fat_percent("HF-3")
