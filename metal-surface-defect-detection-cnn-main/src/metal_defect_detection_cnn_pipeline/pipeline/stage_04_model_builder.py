import sys
from src.metal_defect_detection_cnn_pipeline.utils.custom_logging import logger
from src.metal_defect_detection_cnn_pipeline.utils.custom_exception import CustomException
from src.metal_defect_detection_cnn_pipeline.components.component_04_model_builder import ModelBuilder
from src.metal_defect_detection_cnn_pipeline.config.configuration import ConfigurationManager

class ModelBuilderPipeline:
    def __init__(self):
        pass
    def main(self):
        """Main method to build models using configuration."""
        try:
            # Initialize ConfigurationManager to read and manage configuration files
            logger.info("Initializing ConfigurationManager to fetch configurations.")
            config = ConfigurationManager()

            # Fetch the model builder configuration from ConfigurationManager
            logger.info("Fetching ModelBuilderConfig.")
            model_builder_config = config.model_builder_config()

            # Initialize ModelBuilder class with the fetched configuration
            logger.info("Initializing ModelBuilder class.")
            model_builder = ModelBuilder(config=model_builder_config)

            # Perform Model Building
            logger.info("Starting Model Building process.")
            models_dict = model_builder.build_models()
            return models_dict

        except Exception as e:
            # Log any exceptions that occur during the process
            logger.error("An error occurred during Model Building", exc_info=True)
            raise CustomException(e, sys)


if __name__ == '__main__':
    STAGE_NAME = 'Model Builder Stage'
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = ModelBuilderPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx=============x")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e,sys)