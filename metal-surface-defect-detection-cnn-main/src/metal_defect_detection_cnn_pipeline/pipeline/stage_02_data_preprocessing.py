import sys
from src.metal_defect_detection_cnn_pipeline.utils.custom_logging import logger
from src.metal_defect_detection_cnn_pipeline.utils.custom_exception import CustomException
from src.metal_defect_detection_cnn_pipeline.components.component_02_data_preprocessing import DataPreprocessing
from src.metal_defect_detection_cnn_pipeline.config.configuration import ConfigurationManager

class DataPreprocessingPipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            # Initialize ConfigurationManager to read and manage configuration files
            logger.info("Initializing ConfigurationManager to fetch configurations.")
            config = ConfigurationManager()

            # Fetch the data preprocessing configuration from ConfigurationManager
            logger.info("Fetching DataPreprocessingConfig.")
            data_preprocessing_config = config.get_data_preprocessing_config()

            # Initialize DataPreprocessing class with the fetched configuration
            logger.info("Initializing DataPreprocessing class.")
            data_preprocessing = DataPreprocessing(config=data_preprocessing_config)

            # Perform data preprocessing
            logger.info("Starting data preprocessing process.")
            data_preprocessing.initiate_data_transformer()

        except Exception as e:
            # Log any exceptions that occur during the process
            logger.error("An error occurred during data preprocessing.", exc_info=True)
            raise CustomException(e, sys)


if __name__ == '__main__':
    STAGE_NAME = 'Data Preprocessing Stage'
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = DataPreprocessingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx=============x")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e,sys)