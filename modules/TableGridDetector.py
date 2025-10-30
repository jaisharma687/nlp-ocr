import os
import cv2
import numpy as np

class TableGridDetector:

    def __init__(self, image):
        self.image = image
        self.cells = []

    def execute(self):
        self.grayscale_image()
        self.store_process_image("0_grayscaled.jpg", self.grey)
        self.threshold_image()
        self.store_process_image("1_thresholded.jpg", self.thresholded_image)
        self.invert_image()
        self.store_process_image("2_inverted.jpg", self.inverted_image)
        self.detect_vertical_lines()
        self.store_process_image("3_vertical_lines.jpg", self.vertical_lines)
        self.detect_horizontal_lines()
        self.store_process_image("4_horizontal_lines.jpg", self.horizontal_lines)
        self.combine_lines()
        self.store_process_image("5_grid_structure.jpg", self.grid_structure)
        self.find_grid_intersections()
        self.extract_cells()
        self.visualize_grid()
        self.store_process_image("6_detected_grid.jpg", self.grid_visualization)
        return {
            'grid_structure': self.grid_structure,
            'cells': self.cells,
            'vertical_lines': self.v_lines,
            'horizontal_lines': self.h_lines
        }

    def grayscale_image(self):
        self.grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def threshold_image(self):
        self.thresholded_image = cv2.threshold(self.grey, 127, 255, cv2.THRESH_BINARY)[1]

    def invert_image(self):
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)

    def detect_vertical_lines(self):
        # Detect vertical lines using morphological operations
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        self.vertical_lines = cv2.morphologyEx(self.inverted_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    def detect_horizontal_lines(self):
        # Detect horizontal lines using morphological operations
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        self.horizontal_lines = cv2.morphologyEx(self.inverted_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    def combine_lines(self):
        self.grid_structure = cv2.add(self.vertical_lines, self.horizontal_lines)

    def find_grid_intersections(self):
        # Find contours to get line positions
        contours_v, _ = cv2.findContours(self.vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_h, _ = cv2.findContours(self.horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get sorted vertical line positions (x coordinates)
        v_positions = []
        for cnt in contours_v:
            x, y, w, h = cv2.boundingRect(cnt)
            v_positions.append(x)
        self.v_lines = sorted(set(v_positions))

        # Get sorted horizontal line positions (y coordinates)
        h_positions = []
        for cnt in contours_h:
            x, y, w, h = cv2.boundingRect(cnt)
            h_positions.append(y)
        self.h_lines = sorted(set(h_positions))

    def extract_cells(self):
        # Extract cell coordinates based on grid lines
        for i in range(len(self.h_lines) - 1):
            row_cells = []
            for j in range(len(self.v_lines) - 1):
                cell = {
                    'row': i,
                    'col': j,
                    'x': self.v_lines[j],
                    'y': self.h_lines[i],
                    'width': self.v_lines[j + 1] - self.v_lines[j],
                    'height': self.h_lines[i + 1] - self.h_lines[i]
                }
                row_cells.append(cell)
            self.cells.append(row_cells)

    def visualize_grid(self):
        # Create visualization of detected grid
        self.grid_visualization = self.image.copy()
        
        # Draw vertical lines
        for x in self.v_lines:
            cv2.line(self.grid_visualization, (x, 0), (x, self.image.shape[0]), (0, 255, 0), 2)
        
        # Draw horizontal lines
        for y in self.h_lines:
            cv2.line(self.grid_visualization, (0, y), (self.image.shape[1], y), (0, 255, 0), 2)
        
        # Draw cell numbers
        for row in self.cells:
            for cell in row:
                center_x = cell['x'] + cell['width'] // 2
                center_y = cell['y'] + cell['height'] // 2
                text = f"({cell['row']},{cell['col']})"
                cv2.putText(self.grid_visualization, text, (center_x - 20, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    def store_process_image(self, file_name, image):
        path = "./process_images/table_grid_detector/"
        os.makedirs(path, exist_ok=True)
        file_path = path + file_name
        cv2.imwrite(file_path, image)