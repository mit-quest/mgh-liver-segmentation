import cv2
import math
import numpy as np

def _select_liver_tissue_white_areas(image_path, green, blue, red, liver_tissue_mask):
    """
    TODO: Add docstring

    liver_tissue_mask: 1 if pixel is within liver tissue, 0 otherwise
    """
    img = cv2.imread(image_path, 1)
    num_rows, num_cols, num_channels = img.shape
    identified_liver_tissue = np.zeros((num_rows, num_cols))
    for r in range(num_rows):
        for c in range(num_cols):
            # TODO: Mask is incorrect for many images, so remove liver_tissue_mask
            # if liver_tissue_mask[r][c] > 0 and blue < img[r][c][0] and green < img[r][c][1] and red < img[r][c][2]:
            if blue < img[r][c][0] and green < img[r][c][1] and red < img[r][c][2]:
                identified_liver_tissue[r][c] = 255
    return identified_liver_tissue

def _find_significant_contour(edgeImg):
    contours, hierarchy = cv2.findContours(
        edgeImg,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    # Find level 1 contours
    level1Meta = []
    for contourIndex, tupl in enumerate(hierarchy[0]):
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl.copy(), 0, [contourIndex])
            level1Meta.append(tupl)

    # From among them, find the contours with large surface area.
    contoursWithArea = []
    for tupl in level1Meta:
        contourIndex = tupl[0]
        contour = contours[contourIndex]
        area = cv2.contourArea(contour)
        contoursWithArea.append([contour, area, contourIndex])
    contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
    largestContour = contoursWithArea[0][0]
    return largestContour

def _salt_pepper_noise(edgeImg):
    count = 0
    lastMedian = edgeImg
    median = cv2.medianBlur(edgeImg, 3)
    while not np.array_equal(lastMedian, median):
        zeroed = np.invert(np.logical_and(median, edgeImg))
        edgeImg[zeroed] = 0
        count = count + 1
        if count > 70:
            break
        lastMedian = median
        median = cv2.medianBlur(edgeImg, 3)

def _make_liver_tissue_mask(image_path):
    image_vec = cv2.imread(image_path, 1)
    g_blurred = cv2.GaussianBlur(image_vec, (5, 5), 0)
    blurred_float = g_blurred.astype(np.float32) / 255.0
    edgeDetector = cv2.ximgproc.createStructuredEdgeDetection('edge_detection_model.yml') # TODO: Add file to directory
    edges = edgeDetector.detectEdges(blurred_float) * 255.0

    edges_ = np.asarray(edges, np.uint8)
    _salt_pepper_noise(edges_)

    contour = _find_significant_contour(edges_)
    # Draw the contour on the original image
    contourImg = np.copy(image_vec)
    cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)

    # TODO: Are trimap and trimap_print necessary?
    mask = np.zeros_like(edges_)
    cv2.fillPoly(mask, [contour], 255)
    # calculate sure foreground area by dilating the mask
    mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)
    # mark initial mask as "probably background"
    # and mapFg as sure foreground
    trimap = np.copy(mask)
    trimap[mask == 0] = cv2.GC_BGD
    trimap[mask == 255] = cv2.GC_PR_BGD
    trimap[mapFg == 255] = cv2.GC_FGD
    # visualize trimap
    trimap_print = np.copy(trimap)
    trimap_print[trimap_print == cv2.GC_PR_BGD] = 0 # buffer, probably background
    trimap_print[trimap_print == cv2.GC_FGD] = 255
    # Get only part of cropped image with liver tissue
    mask = trimap_print
    _, mask = cv2.threshold(mask, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    return mask

def _local_max(l):
    return [(i, y) for i, y in enumerate(l)
            if (((i == 0) or (l[i - 1] <= y)) and ((i == len(l) - 1) or (y > l[i+1]))) or
               (((i == 0) or (l[i - 1] < y)) and ((i == len(l) - 1) or (y >= l[i+1])))]

def _moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def _get_moving_average_histogram(image_path, w):
    img = cv2.imread(image_path, 1)
    color = ('b','g') # TODO: Pass in as param, DRY
    mov_average_per_color = dict()
    for i, col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256]).flatten()
        mov_average = _moving_average(histr, w)
        mov_average_per_color[col] = mov_average
    return mov_average_per_color

def _find_peaks(image_path, two_peak=False): # TODO: Change two_peak to required param
    img = cv2.imread(image_path, 1)
    color = ('b','g')
    color_to_peaks = dict()
    mov_average_per_color = _get_moving_average_histogram(image_path, 4)
    for i, col in enumerate(color):
        mov_average = mov_average_per_color[col]
        histr_local_max = _local_max(mov_average)
        histr_local_max = [(val, height) for val, height in histr_local_max if 75 < val < 255]
        histr_local_max.sort(key=lambda x:x[1], reverse=True)

        two_peaks = []
        if two_peak:
            two_peaks.append(histr_local_max[0])
            for i, tup in enumerate(histr_local_max, start=1):
                val, height = tup
                if abs(val - two_peaks[-1][0]) > 10:
                    two_peaks.append(tup)
                    break
            color_to_peaks[col] = [two_peaks[0][0], two_peaks[1][0]] # Add peak val, not height
    return color_to_peaks

def find_white_areas_in_liver_tissue(image_path):
    color_to_peaks = _find_peaks(image_path, two_peak=True)
    min_peaks = []
    max_peaks = []
    color = ('b','g')
    for i, col in enumerate(color):
        min_peaks.append(min(color_to_peaks[col]))
        max_peaks.append(max(color_to_peaks[col]))

    buffer = 5
    green_thresh = min(170, 0.5*min_peaks[1] + 0.5*max_peaks[1] - buffer)
    blue_thresh = min(170, 0.5*min_peaks[0] + 0.5*max_peaks[0] - buffer)
    liver_mask = _make_liver_tissue_mask(image_path)
    white_areas_in_liver_tissue = _select_liver_tissue_white_areas(image_path, green=green_thresh, blue=blue_thresh, red=0, liver_tissue_mask=liver_mask)
    white_areas_in_liver_tissue = white_areas_in_liver_tissue.astype(np.int64)
    return white_areas_in_liver_tissue