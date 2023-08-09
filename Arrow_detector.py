import cv2
import numpy as np

class Arrow_detector:
    def __init__(self, edge_image):
        self.edge_image = edge_image
        self.arrow_contours = []

    def find_arrows(self):
        # Find contours in the edge image
        contours, _ = cv2.findContours(self.edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            # Calculate the perimeter of the contour
            perimeter = cv2.arcLength(contour, True)

            # Approximate the number of vertices in the contour using Ramer-Douglas-Peucker algorithm
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Draw a circle around each contour
            center, radius = cv2.minEnclosingCircle(contour)
            center = tuple(map(int, center))
            radius = int(radius)
            cv2.circle(self.edge_image, center, radius, (0, 255, 0), 2)

            # Check if the contour is arrow-like based on number of vertices and aspect ratio
            if len(approx) == 7 and cv2.contourArea(contour) > 50:  # Modify the criteria as needed
                self.arrow_contours.append(approx)

        print(f"Detected {len(self.arrow_contours)} arrow(s).")

    def draw_arrow_contours(self, image):
        print(str(len(self.arrow_contours)))
        for contour in self.arrow_contours:
            cv2.drawContours(image, [contour], 0, (0, 255, 0), 4)  # Draw green bounding boxes

        return image

    def get_arrow_contours(self):
        return self.arrow_contours
