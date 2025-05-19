import numpy as np
from PIL import Image
import joblib
import sys
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf


from src.metal_defect_detection_cnn_pipeline.utils.custom_logging import logger
from src.metal_defect_detection_cnn_pipeline.utils.custom_exception import CustomException
from src.metal_defect_detection_cnn_pipeline.entity.config_entity import DataPreprocessingConfig

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        """
        Initializes the Preprocessing class with the provided configuration.
        """
        self.config = config
        self.train_mean = None
        self.train_std = None
        self.target_size = self.config.target_size
        logger.info("Initializing DataPreprocessingConfig with the following configuration:")
        for key, value in vars(self.config).items():
            logger.info(f"{key}: {value}")        

    def resize_images(self, input_dir, output_dir):
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            for root, _, files in os.walk(input_dir):
                for file_name in files:
                    if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                        file_path = os.path.join(root, file_name)
                        with Image.open(file_path) as img:
                            img = img.convert("RGB")  # Convert to RGB
                            resized_img = img.resize(self.config.target_size, Image.Resampling.LANCZOS)
                            relative_path = os.path.relpath(root, input_dir)
                            output_subdir = os.path.join(output_dir, relative_path)
                            if not os.path.exists(output_subdir):
                                os.makedirs(output_subdir)
                            output_path = os.path.join(output_subdir, file_name)
                            resized_img.save(output_path)
            logger.info("Resizing complete.")
        except Exception as e:
            logger.error(f"Error in resize_images: {e}")
            raise CustomException(e,sys)

    def compute_normalization_stats(self, train_dataset_dir, is_train=True):
        """Computes mean and std of dataset in a memory-efficient way."""
        try:
            if not is_train:
                logger.info("Skipping normalization computation for non-training data.")
                return

            pixel_sum = 0
            pixel_sq_sum = 0
            num_pixels = 0

            for root, _, files in os.walk(train_dataset_dir):
                for file_name in files:
                    if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                        file_path = os.path.join(root, file_name)
                        with Image.open(file_path) as img:
                            img_array = np.array(img) / 255.0  # Normalize pixels to [0,1]
                            pixel_sum += np.sum(img_array, axis=(0, 1))
                            pixel_sq_sum += np.sum(np.square(img_array), axis=(0, 1))
                            num_pixels += img_array.shape[0] * img_array.shape[1]

            if num_pixels == 0:
                raise ValueError("No valid images found for computing normalization statistics.")

            self.train_mean = pixel_sum / num_pixels
            self.train_std = np.sqrt((pixel_sq_sum / num_pixels) - np.square(self.train_mean))
            logger.info(f"Computed Train Mean: {self.train_mean}, Train Std: {self.train_std}")
        except Exception as e:
            logger.error(f"Error in compute_normalization_stats: {e}")
            raise CustomException(e,sys)

    def normalize_image(self, img):
        try:
            if self.train_mean is None or self.train_std is None:
                raise ValueError("Normalization stats (mean, std) are not computed. Call compute_normalization_stats() first.")
            return (img / 255.0 - self.train_mean) / self.train_std
        except Exception as e:
            logger.error(f"Error in normalize_image: {e}")
            raise CustomException(e,sys)

    def augment_image_for_training(self, image):
        try:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            return image
        except Exception as e:
            logger.error(f"Error in preprocess_for_training: {e}")
            raise CustomException(e,sys)

    def save(self, filename):
        """Save the preprocessor object using joblib."""
        try:
            joblib.dump(self, filename)
            logger.info(f"Preprocessor saved to {filename}")
        except Exception as e:
            logger.error(f"Error in saving preprocessor: {e}")
            raise CustomException(e,sys)


    def list_image_counts(self, base_dir):
        """Counts and prints the number of images in each class folder."""
        if not base_dir:
            logger.error(f"Dataset directory {base_dir} not exist")
            return

        for root, _, files in os.walk(base_dir):
            # Filter image files based on common formats
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
            if image_files:  # Display only folders containing images
                logger.info(f"Folder: {os.path.relpath(root, base_dir)} | Number of images: {len(image_files)}")


    def initiate_data_transformer(self):
        """Runs the full preprocessing pipeline."""
        try:
            # Resize images in the train, validation, and test datasets
            self.resize_images(self.config.raw_train, self.config.resize_train)
            self.resize_images(self.config.raw_valid, self.config.resize_valid)
            self.resize_images(self.config.raw_test, self.config.resize_test)


            base_dir = os.path.dirname(os.path.dirname(self.config.resize_train))
            self.list_image_counts(base_dir)

            # Compute normalization statistics
            self.compute_normalization_stats(train_dataset_dir=self.config.resize_train, is_train=True)

            # Save the preprocessor object
            self.save(self.config.preprocessor_path)
        except Exception as e:
            logger.error(f"Error in initiate_data_transformer: {e}")
            raise CustomException(e,sys)