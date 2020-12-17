def filter_nonfat_by_size(image_name, original_folder, binary_folder=None, inv_binary_folder=None, min_size=10, max_size=1000):
    binary = convert_single_image_to_binary(image_name, original_folder, binary_folder=binary_folder)
    binary = binary_opening(binary, disk(1))
    np_graph = binary
    row, col = np_graph.shape
    graph = np_graph.tolist()
    g = Graph(row, col, graph)
    islands = g.findIslands()

    # Filter islands by size. Keep islands with sizes in [min_size, max_size)
    island_to_score_map = dict()
    for island in islands:
        if min_size <= len(island) < max_size:
            island_to_score_map[tuple(island)] = 1 # Score of 1 means keep
        else:
            island_to_score_map[tuple(island)] = 0 # Score of 0 means do not keep

    # Create binary image without obvious non-fat. Color black non-fat islands
    binary_without_obvious_non_fat = np.zeros(binary.shape)
    for island, score in island_to_score_map.items():
        if score == 1:
            for x, y in island:
                binary_without_obvious_non_fat[x][y] = 255
    binary_without_obvious_non_fat = binary_without_obvious_non_fat.astype(np.uint8)

    # Convert to inverted binary
    inverted_binary = 255 - binary_without_obvious_non_fat

    # Save inverted binary image
    if inv_binary_folder is not None:
        cv2.imwrite(f"{inv_binary_folder}/{image_name}", inverted_binary)

image_names = ['HF-3 formalin - pre - 20x - a - 1.tiff',
               'HF-10 formalin - 6h - 20x - f - 1.tiff',
               'HF-21 formalin - 3h - 20x - g - 1.tiff']
for image_name in image_names:
    filter_nonfat_by_size(image_name,
                        'drive/MyDrive/ImageJ-original-images',
                        binary_folder='drive/MyDrive/ImageJ-binary-images',
                        inv_binary_folder='drive/MyDrive/ImageJ-inv-binary-images',
                        min_size=10, max_size=1000)
