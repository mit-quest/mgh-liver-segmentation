import cv2
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
