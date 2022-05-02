import cv2
import numpy as np

SHARPEN_FILTER = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

def sharpen_image(image_path):
    original_image = cv2.imread(image_path)
    sharpened_image = cv2.filter2D(original_image, -1, SHARPEN_FILTER)
    return original_image, sharpened_image