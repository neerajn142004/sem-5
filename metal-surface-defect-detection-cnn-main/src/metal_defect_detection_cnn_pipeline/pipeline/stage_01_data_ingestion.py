import sys
from src.metal_defect_detection_cnn_pipeline.utils.custom_logging import logger
from src.metal_defect_detection_cnn_pipeline.utils.custom_exception import CustomException
from src.metal_defect_detection_cnn_pipeline.components.component_01_data_ingestion import DataIngestion
from src.metal_defect_detection_cnn_pipeline.config.configuration import ConfigurationManager


class DataIngestionPipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            # Initialize the ConfigurationManager to load the configuration files.
            # This includes YAML files that define the paths, URLs, and other parameters.
            config = ConfigurationManager()

            # Get the data ingestion configuration object, which contains paths and URL details.
            # This configuration will be used for downloading the data and file extraction.
            data_ingestion_config = config.get_data_ingestion_config()

            # Create an instance of the DataIngestion class with the loaded configuration.
            # This class is responsible for downloading the file and extracting it (if needed).
            data_ingestion = DataIngestion(config=data_ingestion_config)

            # Call the download_file method to download the file from the specified URL.
            # If the file already exists locally, it will simply log the size and skip downloading.
            data_ingestion.setup_dataset()

        except Exception as e:
            # If any exception occurs during the process, raise a CustomException.
            # This will include the original exception message and traceback for easier debugging.
            raise CustomException(e, sys)

if __name__ == '__main__':
    STAGE_NAME = 'Data Ingestion Stage'
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx=============x")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e,sys)