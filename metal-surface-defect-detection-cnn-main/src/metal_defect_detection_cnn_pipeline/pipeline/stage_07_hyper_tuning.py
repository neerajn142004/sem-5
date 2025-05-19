import sys
from src.metal_defect_detection_cnn_pipeline.utils.custom_logging import logger
from src.metal_defect_detection_cnn_pipeline.utils.custom_exception import CustomException
from src.metal_defect_detection_cnn_pipeline.components.component_07_hyper_tuning import ModelTuner
from src.metal_defect_detection_cnn_pipeline.config.configuration import ConfigurationManager

class ModelHyperparameterTuningPipeline:
    """
    Pipeline class to manage the model training process. 
    It initializes the configuration, sets up the model trainer, and executes the training process.
    """

    def __init__(self):
        """
        Initializes the ModelTrainerTrainingPipeline class. 
        Currently, there are no specific attributes to initialize.
        """
        pass

    def main(self):
        """
        Main method to execute the model training pipeline. 
        This method:
        - Reads configuration using the ConfigurationManager.
        - Initializes the ModelTrainer with the required configurations.
        - Executes the training process.
        """
        try:
            logger.info("Initializing ConfigurationManager for model trainer.")
            # Initialize configuration manager
            config = ConfigurationManager()

            # Fetch the model trainer configuration
            logger.info("Fetching model trainer configuration.")
            hyperpameter_tuning_config = config.get_hyperpameter_tuning_config()


            # Initialize the model trainer with the configuration
            logger.info("Initializing ModelTrainer with the fetched configuration.")
            model_tuner = ModelTuner(hyperpameter_tuning_config)

            # Start the training process
            logger.info("Starting model training.")
            model_tuner.initiate_hyper_tuning()
            logger.info("Model training completed successfully.")

        except Exception as e:
            # Log any exceptions that occur during the training process
            logger.error("An error occurred during the Model Trainer Pipeline execution.", exc_info=True)
            raise CustomException(e, sys)


if __name__ == '__main__':
    STAGE_NAME = 'Model Hyperparameter Tuning Stage'

    try:
        # Log the start of the stage
        logger.info(f">>>>>>> Stage: {STAGE_NAME} started <<<<<<<")

        # Create the pipeline object and execute the main method
        obj = ModelHyperparameterTuningPipeline()
        obj.main()

        # Log the successful completion of the stage
        logger.info(f">>>>>>> Stage: {STAGE_NAME} completed <<<<<<<\n\nx=============x")

    except Exception as e:
        # Log and re-raise any exceptions during the pipeline execution
        logger.exception(f"An error occurred in stage: {STAGE_NAME}.")
        raise CustomException(e, sys)