import concurrent.futures
import cv2
import math
import numpy as np
from skimage.morphology import binary_opening, disk, remove_small_holes, remove_small_objects, binary_erosion

import frozen_only

class Graph:
    def __init__(self, row, col, graph):
        self.ROW = row
        self.COL = col
        self.graph = graph

    def isSafe(self, i, j, visited):
        # Row number is in range, column number is in range, value is 1 and not yet visited
        return 0 <= i and i < self.ROW and 0 <= j and j < self.COL and not visited[i][j] and self.graph[i][j]

    def BFS(self, i, j, visited, island):
        # Utility function to do BFS for a 2D boolean matrix. Uses only the 4 neighbors as adjacent vertices
        rowNbr = [-1, 0, 1, 0]
        colNbr = [0, -1, 0, 1]
        q = []
        q.append((i,j))
        visited[i][j] = True

        while len(q) != 0:
            x,y = q.pop(0)
            for k in range(len(rowNbr)):
                if self.isSafe(x + rowNbr[k], y + colNbr[k], visited):
                    island.append((x + rowNbr[k], y + colNbr[k]))
                    visited[(x) + rowNbr[k]][y + colNbr[k]] = True
                    q.append((x + rowNbr[k], y + colNbr[k]))

    def findIslands(self):
        # Make a bool array to mark visited cells. Initially all cells are unvisited
        visited = [[False for j in range(self.COL)]for i in range(self.ROW)]
        # Initialize count as 0 and traverse through cells of given matrix
        index = 0
        islands = []
        for i in range(self.ROW):
            for j in range(self.COL):
                # If a cell with value 1 is not visited yet, then new island found
                if visited[i][j] == False and self.graph[i][j] == 1:
                    # Visit all cells in this island and increment island count
                    island = []
                    self.BFS(i, j, visited, island)
                    islands.append(island)
                    index += 1
        return islands

def new_graph(np_graph):

    row, col = np_graph.shape
    graph = np_graph.tolist()
    g = Graph(row, col, graph)
    islands = g.findIslands()

    return islands

# Assign number between 0 (most likely non-fat) and 1 (most likely fat) for
# probability that island is fat. Input is list of lists of pixels for each island
def get_fat_score_for_watershed_image(islands, original_mask, circularity_threshold=0.5, contour_area_vs_perimeter=True, min_size=10, max_size=1000, is_frozen=False):
    # Filter islands. Map of tuple of (x, y) coordinates of island to
    # probability that island is fat, initialized to 0.5 (undecided)
    island_to_score_map = dict()
    for island in islands:
        if min_size <= len(island) < max_size: # Filter out small and large tears and holes
            island_to_score_map[tuple(island)] = 0.5
        else:
            island_to_score_map[tuple(island)] = 0 # Small and large tears and holes are assumed to be non-fat (0)

    # --- Assign circularity to islands ---
    island_to_circularity_map = dict() # Map of tuple of (x, y) coordinates of island to circularity score
    for island in island_to_score_map:
        if island_to_score_map[island] > 0:
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
            center, radius = cv2.minEnclosingCircle(contour)

            if contour_area_vs_perimeter:
                # --- Method to calculate circularity: contour area vs. contour perimeter ---
                circularity = (2 * math.pi * math.sqrt(contour_area / math.pi)) / contour_perimeter
            else:
                # --- Method to calculate circularity: contour area vs. minimum enclosing circle area ---
                min_enclosing_circle_area = math.pi * radius * radius
                circularity = contour_area / min_enclosing_circle_area

            # Assign circularity
            island_to_circularity_map[island] = circularity

    # --- Filter islands by circularity ---
    for island, circularity in island_to_circularity_map.items():
        if circularity >= circularity_threshold:
            island_to_score_map[island] = 1
        if circularity < circularity_threshold:
            island_to_score_map[island] = 0

    # Create new mask
    new_mask = np.zeros(original_mask.shape)
    for island, circularity in island_to_circularity_map.items():
        if island_to_score_map[island] == 1: # 0 means non-fat, 0.5 means unsure, 1 means fat
            for x, y in island:
                new_mask[x][y] = 255

    new_mask = new_mask.astype(np.uint8)
    return new_mask

def prepare_image(image_path, is_frozen):
    if is_frozen:
        white_areas_in_liver_tissue = frozen_only.find_white_areas_in_liver_tissue(image_path)
        bool_input = white_areas_in_liver_tissue
    else:
        # Find sharpened, grayscale, and binary images
        original_image, sharpened_image = sharpen_image(image_path)
        sharpened_gray = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
        _, sharpened_binary = cv2.threshold(sharpened_gray, 195, 255, cv2.THRESH_BINARY)
        bool_input = sharpened_binary

    # Remove noise
    opened_image_bool = remove_small_holes(bool_input, area_threshold=2) # Change black area < 5 pixels to white
    opened_image_bool = remove_small_objects(opened_image_bool, min_size=10) # Change white area < 10 pixels to black
    opened_image = opened_image_bool.astype(np.uint8)  # Convert to an unsigned byte
    opened_image *= 255

    # Apply erosion
    erode_image_bool = binary_erosion(opened_image_bool)
    erode_image = erode_image_bool.astype(np.uint8)
    erode_image *= 255

    return erode_image_bool
