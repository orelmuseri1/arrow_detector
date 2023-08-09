import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

def darken_specific_colors(image, color_ranges, factor):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Iterate through the specified color ranges
    for color_range in color_ranges:
        lower_range, upper_range = color_range

        # Create a mask for the specified color range
        mask = cv2.inRange(hsv_image, lower_range, upper_range)

        # Adjust the value channel to make the colors darker
        hsv_image[:, :, 2] = (hsv_image[:, :, 2] * (1 - mask) + mask * factor).astype(np.uint8)

    # Convert the modified HSV image back to BGR color space
    modified_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return modified_image

def enhance_image_quality(image):
    # Apply Gaussian blur for noise reduction
    blurred_image1 = cv2.GaussianBlur(image, (5, 5), 1)
    blurred_image = cv2.medianBlur(blurred_image1,5)

    # Increase contrast and brightness
    enhanced_image = cv2.convertScaleAbs(blurred_image1, alpha=1.9, beta=30)
    # display_cv2_image(enhanced_image, "enhanced_image2")

    # Define the color ranges for green, yellow, and orange in HSV color space
    green_range = (np.array([40, 40, 40]), np.array([80, 255, 255]))
    yellow_range = (np.array([20, 40, 40]), np.array([40, 255, 255]))
    orange_range = (np.array([5, 40, 40]), np.array([20, 255, 255]))
    red_range = (np.array([0, 40, 40]), np.array([5, 255, 255]))

    # Combine the color ranges
    color_ranges = [red_range,green_range, yellow_range, orange_range]

    # Darken the specified colors in the image
    darkened_image = darken_specific_colors(enhanced_image, color_ranges, factor=1.1)
    display_cv2_image(darkened_image, "gray_image")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(darkened_image, cv2.COLOR_BGR2GRAY)
    # Apply unsharp masking for sharpening
    sharpened_image = cv2.addWeighted(gray_image, 1.1, cv2.GaussianBlur(gray_image, (0, 0), 2), -0.5, 0)

    return sharpened_image


def display_cv2_image(image, title):
    # Convert the BGR image to RGB format for Matplotlib
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Create a figure with a single subplot
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 1, 1)
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis('off')

    # Display the image
    plt.show()


class PreProcessor:
    def __init__(self, image):
        self.image = image
        self.preprocessed_image = None

    def display_image(self, image):
        # Display the dilated edges
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 1, 1)
        plt.imshow(self.image, cmap='gray')
        plt.title('display func image:')
        plt.axis('off')
        plt.show()

    def preprocess_image(self):
        display_cv2_image(self.image, "origin")
        enhance_image_qualityvar = enhance_image_quality(self.image)
        display_cv2_image(enhance_image_qualityvar, "enhance_image_qualityvar(self.image)")

        # Convert the image to grayscale
        #grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # display_cv2_image(grayscale_image,"grayscale_image")

        #blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
        # display_cv2_image(blurred_image,"blurred_image")

        self.preprocessed_image = enhance_image_qualityvar

        # Apply Canny edge detection
        edges = cv2.Canny(enhance_image_qualityvar, 200, 300)
        display_cv2_image(edges, "edges")

        # Apply dilation and erosion
        kernel = np.ones((3, 3), np.uint8)

        # Apply dilation and erosion
        dilated_image = cv2.dilate(edges, kernel, iterations=1)

        eroded_image = cv2.erode(edges, kernel, iterations=1)
        dilated_image2 =cv2.erode(dilated_image, kernel, iterations=1)

        # Display the original, dilated, and eroded images
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(edges, cmap='gray')
        plt.title('Original Binary Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(dilated_image, cmap='gray')
        plt.title('Dilated Image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(eroded_image, cmap='gray')
        plt.title('Eroded Image')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


        # Find contours and hierarchy
        contours, hierarchy = cv2.findContours(dilated_image2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the original image for visualization
        visualized_image = cv2.cvtColor(dilated_image2, cv2.COLOR_GRAY2BGR)
        # Loop through contours and hierarchy
        for idx, contour in enumerate(contours):
            parent_idx = hierarchy[0][idx][3]  # Parent index in the hierarchy
            area = cv2.contourArea(contour)
            # Check if the area is within the specified range
            if parent_idx != -1 and 150 <= area <= 600:
                # This contour has a parent and falls within the area range, indicating it's a hole
                # Create a new image for the blob with a hole
                blob_with_hole = np.zeros_like(visualized_image)
                cv2.drawContours(blob_with_hole, [contour], -1, 255, -1)

                # Display the blob with a hole
                plt.imshow(blob_with_hole, cmap='gray')
                plt.title(f'Blob1 with Hole {idx}')
                plt.axis('off')
                plt.show()

#        # Loop through contours and hierarchy
#        for idx, contour in enumerate(contours):
#            parent_idx = hierarchy[0][idx][3]  # Parent index in the hierarch   y
#            if parent_idx != -1:
#                # This contour has a parent, indicating it's a hole
#                cv2.drawContours(dilated_image2, [contour], -1, (0, 255, 0), 2)

        # Display the visualized image
#        plt.imshow(cv2.cvtColor(dilated_image2, cv2.COLOR_BGR2RGB))
#        plt.title('Blobs with Holes end ')
#        plt.axis('off')
#        plt.show()
        return self.preprocessed_image

