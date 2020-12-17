def convert_single_image_to_binary(image_name, original_folder, binary_folder=None, show_images=False):
    image = cv2.imread(f"{original_folder}/{image_name}")
    original = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY) # 200 was experimentally determined

    if show_images:
        cv2_imshow(original)
        cv2_imshow(binary)

    # Save binary image without binary opening
    if binary_folder is not None:
        cv2.imwrite(f"{binary_folder}/{image_name}", binary)

    return binary

def convert_multiple_images_to_binary(image_names, original_folder, binary_folder=None, show_images=False):
    for image_name in image_names:
        convert_single_image_to_binary(image_name, original_folder, binary_folder=binary_folder, show_images=show_images)

image_names = ['HF-3 formalin - pre - 20x - a - 1.tiff',
               'HF-10 formalin - 6h - 20x - f - 1.tiff',
               'HF-21 formalin - 3h - 20x - g - 1.tiff']
convert_multiple_images_to_binary(image_names,
                                  'drive/MyDrive/ImageJ-original-images',
                                  binary_folder='drive/MyDrive/ImageJ-binary-images',
                                  show_images=False)
