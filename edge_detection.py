import cv2
import numpy as np
from PIL import Image
import common
import copy


def canny_edge_detection(image_path, save_path, blur_kernel_size, canny_threshold1, canny_threshold2, morph_kernel_size):
    # Load original image in grayscale
    image = cv2.imread(image_path, 0)
    pil_image = Image.fromarray(image)
    pil_image.save("/Users/katexu/Downloads/liver_images/grayscale.tiff")

    # Blur image
    blur_image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)

    # Find edges using Canny algorithm
    edge_image = cv2.Canny(blur_image, canny_threshold1, canny_threshold2)

    # Try morphological closing (dilation followed by erosion)
    kernel = np.ones((morph_kernel_size, morph_kernel_size),np.uint8)
    closed_image = cv2.morphologyEx(edge_image, cv2.MORPH_CLOSE, kernel)

    # Save image
    save_image = Image.fromarray(closed_image)
    save_image.save(save_path)

    return closed_image


# Example (tweak the parameters)
image_path = "/Users/katexu/Downloads/liver_images/Frozen-HF-27-dh H&E - 20x - b - 1.tiff"
edge_detection_save_path = "/Users/katexu/Downloads/liver_images/edge_detection.tiff"
blur_kernel_size = 5
canny_threshold1 = 50
canny_threshold2 = 100
morph_kernel_size = 7
canny_image = canny_edge_detection(image_path, edge_detection_save_path, blur_kernel_size, canny_threshold1, canny_threshold2, morph_kernel_size)


def find_large_areas_in_canny_image(canny_image, large_area_threshold, save_path):
    # Find islands in canny image
    canny_image = 255 - canny_image
    canny_image = np.array(canny_image, dtype=bool)
    num_rows, num_cols = canny_image.shape
    image_of_large_areas = np.zeros((num_rows, num_cols))
    canny_graph = canny_image.tolist()
    canny_g = common.Graph(num_rows, num_cols, canny_graph)
    canny_islands = canny_g.findIslands()

    # Isolate large islands
    for island in canny_islands:
        if len(island) >= large_area_threshold:
            for row, col in island:
                image_of_large_areas[row][col] = 255
        else:
            for row, col in island:
                image_of_large_areas[row][col] = 0

    # Save image
    save_image_of_large_areas = Image.fromarray(image_of_large_areas)
    save_image_of_large_areas.save(save_path)

    return image_of_large_areas


# Example (tweak the parameters)
large_area_threshold = 1000
large_area_save_path = "/Users/katexu/Downloads/liver_images/large_areas.tiff"
image_of_large_areas = find_large_areas_in_canny_image(canny_image, large_area_threshold, large_area_save_path)
