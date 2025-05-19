import sys
from src.metal_defect_detection_cnn_pipeline.utils.custom_logging import logger
from src.metal_defect_detection_cnn_pipeline.utils.custom_exception import CustomException
from src.metal_defect_detection_cnn_pipeline.components.component_03_dataset_loader import DatasetLoader
from src.metal_defect_detection_cnn_pipeline.config.configuration import ConfigurationManager

class DatasetLoaderPipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            # Initialize ConfigurationManager to read and manage configuration files
            logger.info("Initializing ConfigurationManager to fetch configurations.")
            config = ConfigurationManager()

            # Fetch the dataset loader configuration from ConfigurationManager
            logger.info("Fetching DatasetLoaderConfig.")
            dataset_loader_config = config.get_dataset_loader_config()

            # Initialize DatasetLoader class with the fetched configuration
            logger.info("Initializing DatasetLoader class.")
            dataset_loader = DatasetLoader(config=dataset_loader_config)

            # Perform dataset loading
            logger.info("Starting dataset loading process.")
            dataset_loader.initiate_dataset_loader()

        except Exception as e:
            # Log any exceptions that occur during the process
            logger.error("An error occurred during dataset loader", exc_info=True)
            raise CustomException(e, sys)


if __name__ == '__main__':
    STAGE_NAME = 'Dataset Loader Stage'
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = DatasetLoaderPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx=============x")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e,sys)