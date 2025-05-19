import os
import sys
import shutil
import time

import kagglehub

from src.metal_defect_detection_cnn_pipeline.utils.custom_logging import logger
from src.metal_defect_detection_cnn_pipeline.utils.custom_exception import CustomException

from src.metal_defect_detection_cnn_pipeline.entity.config_entity import DataIngestionConfig

class DataIngestion:
    """
    DataIngestion handles the process of downloading a file from a URL and extracting
    its contents if it's a ZIP file. This class uses the configuration provided
    during initialization for the download location, source URL, and extraction path.
    """
    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the DataIngestion object with the provided configuration.

        Parameters:
        - config: A DataIngestionConfig object that contains configuration for file download and extraction.
        """
        self.config = config
        self.extract_dir = os.path.join(self.config.root_dir, self.config.input_data_folder)
        self.base_dir = None
        logger.info("DataIngestion object created with the provided configuration.")

        # Log the configuration details at initialization
        logger.info("Initializing DataIngestionConfig with the following configuration:")
        for key, value in vars(self.config).items():
            logger.info(f"{key}: {value}")


    def download_dataset(self):
        """Downloads the dataset using kagglehub and stores the download path."""
        try:
            self.kaggle_download_path = kagglehub.dataset_download(self.config.dataset_identifier)
            logger.info(f"Dataset downloaded to: {self.kaggle_download_path}")
        except Exception as e:
            # Log and raise any exceptions encountered during file download
            logger.error(f"Failed to download file from {self.config.dataset_identifier}. Error: {e}", exc_info=True)
            raise CustomException(e, sys)

    def extract_and_rename(self):
        """
        Extracts the dataset, moves it to the target directory, and renames it.
        """
        # Ensure the extraction directory exists
        if not os.path.exists(self.extract_dir):
            os.makedirs(self.extract_dir)

        try:
            # Move dataset contents to the extraction directory
            for item in os.listdir(self.kaggle_download_path):
                item_path = os.path.join(self.kaggle_download_path, item)
                if os.path.isdir(item_path):  # Process directories only
                    dest_path = os.path.join(self.extract_dir, item)
                    if os.path.exists(dest_path):  # Remove existing destination directory if present
                        shutil.rmtree(dest_path)
                    shutil.copytree(item_path, dest_path)            

            # Define original and new dataset directory paths
            base_dir = os.path.join(self.extract_dir, "NEU Metal Surface Defects Data")
            new_base_dir = os.path.join(self.extract_dir, self.config.new_dir_name)
            
            # Rename the dataset folder for consistency
            if os.path.exists(base_dir):
                if os.path.exists(new_base_dir):  # Remove any existing directory with the new name
                    shutil.rmtree(new_base_dir)
                time.sleep(1)  # Prevent potential file locks before renaming
                shutil.move(base_dir, new_base_dir)
                logger.info(f"Directory renamed to: {new_base_dir}")
            else:
                logger.error(f"Source directory not found: {base_dir}")

            # Update base directory path
            self.base_dir = new_base_dir

        except Exception as e:
            print(f"Error renaming directory: {e}")

    def list_image_counts(self):
        """Counts and prints the number of images in each class folder."""
        if not self.base_dir:
            logger.error("Dataset directory not set. Ensure extraction and renaming is complete.")
            return

        for root, _, files in os.walk(self.base_dir):
            # Filter image files based on common formats
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
            if image_files:  # Display only folders containing images
                logger.info(f"Folder: {os.path.relpath(root, self.base_dir)} | Number of images: {len(image_files)}")

    def setup_dataset(self):
        """Executes all steps: downloading, extracting, renaming, and counting images."""
        self.download_dataset()
        self.extract_and_rename()
        self.list_image_counts()