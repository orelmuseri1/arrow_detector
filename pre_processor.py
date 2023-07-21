import cv2
import matplotlib.pyplot as plt
import os

class PreProcessor:
    def __init__(self, image):
        self.image = image
        self.preprocessed_image = None

    def preprocess_image(self):
        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise (you can adjust the kernel size as needed)
        blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

        self.preprocessed_image = blurred_image

    def display_images(self):
        # Display the original and preprocessed images (optional)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(self.preprocessed_image, cmap='gray')
        plt.title('Preprocessed Image')
        plt.axis('off')

        plt.show()

    def save_preprocessed_image(self, output_dir, filename):
        if self.preprocessed_image is not None:
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, self.preprocessed_image)
            print(f"Preprocessed image saved at: {output_path}")
        else:
            print("Error: Preprocessed image not available.")


