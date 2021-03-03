# Old folder paths before moving to Colab
save_folder = "bucket/new-masks-20x"

# all_liver_files is a list of file paths to original images
all_liver_files = ['drive/MyDrive/HF-21 formalin - 3h - 20x - g - 1.tiff']
for file_name in all_liver_files:
    print("File name: ", file_name)
    image = cv2.imread(file_name)

    # Convert original image to grayscale and binary
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.where(gray<=100, 0, gray)                          # Turns pixel to black if below threshold. Threshold was determined experimentally
    gray = np.where(gray>=205, 255, gray)                        # Turns pixel to white if above threshold. Threshold was determined experimentally
    _, binary = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY) # Converts to binary. Threshold was determined experimentally
    binary = binary_opening(binary, disk(1))                     # “Opens” up (dark) gaps between (bright) features

    # Find islands in binary image
    np_graph = binary
    row, col = np_graph.shape
    graph = np_graph.tolist()
    g = Graph(row, col, graph)
    islands = g.findIslands()

    # Create new mask
    # new_mask = get_fat_score(islands, np_graph, min_size=10, max_size=1500) # Aimed at 20x images in "well-defined fat" category
    # new_mask = get_fat_score(islands, np_graph, min_size=2, max_size=200) # Aimed at 20x images in "micro fat" category
    new_mask = cv2.imread('drive/MyDrive/new-masks-20x_HF-21 formalin - 3h - 20x - g - 1 copy.tiff') # Display an existing new mask

    # Plot 5 images: original, grayscale, binary, new mask, and overlay
    # fig, axes = plt.subplots(1, 5, figsize=(30, 30))
    # ax = axes.ravel()
    # ax[0].imshow(image)
    # ax[0].set_title("Original")
    # ax[1].imshow(gray, cmap=plt.cm.gray)
    # ax[1].set_title("Grayscale")
    # ax[2].imshow(binary, cmap=plt.cm.gray)
    # ax[2].set_title("Binary")
    # ax[3].imshow(new_mask, cmap=plt.cm.gray)
    # ax[3].set_title("New Mask")
    # ax[4].imshow(new_mask, cmap=plt.cm.gray)
    # ax[4].imshow(image, alpha=0.6)
    # ax[4].set_title("Overlay")

    # Plot 3 images: original, new mask, and overlay
    fig, axes = plt.subplots(1, 3, figsize=(30, 30))
    ax = axes.ravel()
    ax[0].imshow(image)
    ax[0].set_title("Original")
    ax[1].imshow(new_mask, cmap=plt.cm.gray)
    ax[1].set_title("New Mask")
    ax[2].imshow(new_mask, cmap=plt.cm.gray)
    ax[2].imshow(image, alpha=0.6)
    ax[2].set_title("Overlay")

    # Save new mask. Old code before moving to Colab
    # file_path_split = file.split("/")
    # save_path = save_folder + "/" + file_path_split[2]
    # cv2.imwrite(save_path, new_mask)
    # plt.close(fig)
