import os
import cv2
from pre_processor import PreProcessor

def main():
    image_path = "regular_arrow.jpeg"
    loaded_image = cv2.imread(image_path)
    preprocessor = PreProcessor(loaded_image)
    preprocessor.preprocess_image()

    # Save preprocessed image to a new directory
    output_directory = "pre_processed_images"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    filename = os.path.splitext(os.path.basename(image_path))[0]
    preprocessed_filename = filename + "_preprocessed.jpg"
    preprocessor.save_preprocessed_image(output_directory, preprocessed_filename)

    preprocessor.display_images()

if __name__ == "__main__":
    main()