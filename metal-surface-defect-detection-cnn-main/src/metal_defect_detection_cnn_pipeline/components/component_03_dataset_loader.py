import joblib
import json
import os
import sys

# Disable oneDNN optimizations for TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

from src.metal_defect_detection_cnn_pipeline.utils.custom_logging import logger
from src.metal_defect_detection_cnn_pipeline.utils.custom_exception import CustomException
from src.metal_defect_detection_cnn_pipeline.entity.config_entity import DatasetLoaderConfig

class DatasetLoader:
    """
    A utility class for loading image datasets, normalizing images using precomputed statistics,
    and extracting class labels.

    Features:
    - Loads image datasets from a specified directory.
    - Normalizes images using precomputed mean and standard deviation.
    - Extracts class names and saves them to a JSON file.
    - Supports batch loading and resizing.
    """

    def __init__(self, config: DatasetLoaderConfig):
        """
        Initializes the DatasetLoader class with the provided configuration.
        """
        try:
            self.config = config
            logger.info("Initializing DatasetLoaderConfig with the following configuration:")
            for key, value in vars(self.config).items():
                logger.info(f"{key}: {value}")
            
            # Load preprocessor object containing normalization parameters
            self.preprocessor = joblib.load(self.config.preprocessor_path)
            self.train_mean = self.preprocessor.train_mean
            self.train_std = self.preprocessor.train_std
            self.class_names = None
        except Exception as e:
            logger.error(f"Error initializing DatasetLoader: {e}")
            raise CustomException(e, sys)

    def normalize(self, img):
        """Normalize an image using precomputed mean and standard deviation."""
        try:
            return (img - self.train_mean) / self.train_std
        except Exception as e:
            logger.error(f"Error in image normalization: {e}")
            raise CustomException(e, sys)

    def load_dataset(self, directory, is_training=False):
        """
        Load dataset from the given directory, extract class names if not already set, and apply normalization.
        Data augmentation is applied only to the training dataset.
        """
        try:
            dataset = image_dataset_from_directory(
                directory,
                batch_size=self.config.batch_size,
                image_size=self.config.target_size,
                shuffle=self.config.shuffle if is_training else False  # Shuffle only for training data
            )
            logger.info(f"Dataset loaded from {directory} with {len(dataset)} batches.")

            # Extract and save class names only once
            if self.class_names is None:
                self.class_names = dataset.class_names
                self.save_class_names(self.class_names)
                logger.info(f"Class names extracted: {self.class_names}")

            # Define preprocessing function to normalize images
            def preprocess(image, label):
                try:
                    image = tf.cast(image, tf.float32) / 255.0  # Scale pixel values to [0,1]
                    image = self.normalize(image)  # Normalize using train mean & std
                    if is_training:
                        image = self.preprocessor.augment_image_for_training(image)  # Apply augmentation
                    return image, label
                except Exception as e:
                    logger.error(f"Error in preprocessing function: {e}")
                    raise CustomException(e, sys)

            return dataset.map(preprocess)
        except Exception as e:
            logger.error(f"Error loading dataset from {directory}: {e}")
            raise CustomException(e, sys)

    def save_class_names(self, class_names):
        """Save extracted class names to a JSON file."""
        try:
            with open(self.config.class_labels_path, "w") as f:
                json.dump(class_names, f)
            logger.info(f"Class names saved to {self.config.class_labels_path}")
        except Exception as e:
            logger.error(f"Error saving class names: {e}")
            raise CustomException(e, sys)

    def initiate_dataset_loader(self):
        """
        Load training, validation, and test datasets, then save them.
        """
        try:
            logger.info("Starting dataset loading process...")
            train_ds = self.load_dataset(self.config.resize_train, is_training=True)
            valid_ds = self.load_dataset(self.config.resize_valid, is_training=False)
            test_ds = self.load_dataset(self.config.resize_test, is_training=False)
            
            # Save datasets
            train_ds.save(str(self.config.train_ds_path))
            logger.info(f"Training dataset saved at {self.config.train_ds_path}")
            valid_ds.save(str(self.config.valid_ds_path))
            logger.info(f"Validation dataset saved at {self.config.valid_ds_path}")
            test_ds.save(str(self.config.test_ds_path))
            logger.info(f"Test dataset saved at {self.config.test_ds_path}")
            
            # Log extracted class labels
            logger.info(f"Class labels mapping: {self.class_names}")
        except Exception as e:
            logger.error(f"Error in dataset initiation: {e}")
            raise CustomException(e, sys)