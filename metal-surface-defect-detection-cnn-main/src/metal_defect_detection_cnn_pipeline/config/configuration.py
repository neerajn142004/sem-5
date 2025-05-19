import sys
import os
from pathlib import Path

from src.metal_defect_detection_cnn_pipeline.constants.constant_values import (
    config_yaml_filepath, 
    schema_yaml_filepath, 
    params_yaml_filepath
)

from src.metal_defect_detection_cnn_pipeline.utils.custom_logging import logger
from src.metal_defect_detection_cnn_pipeline.utils.custom_exception import CustomException
from src.metal_defect_detection_cnn_pipeline.utils.common import read_yaml, create_directories
from src.metal_defect_detection_cnn_pipeline.entity.config_entity import (DataIngestionConfig,
                                                                 DataPreprocessingConfig,
                                                                 DatasetLoaderConfig,
                                                                 ModelBuilderConfig,
                                                                 PrepareCallbacksConfig,
                                                                 ModelTrainerEvaluationConfig,
                                                                 HyperparameterTuningConfig
                                                                 )

class ConfigurationManager:
    """
    A class to manage configuration files and provide configurations for 
    various stages of the pipeline. It includes functionality for reading 
    YAML files, creating necessary directories, and returning configuration 
    objects for different components.
    """

    def __init__(self,
                 config_filepath=config_yaml_filepath,
                 schema_filepath=schema_yaml_filepath,
                 params_filepath=params_yaml_filepath):
        """
        Initializes the ConfigurationManager by loading YAML files and ensuring
        required directories are created.
        """
        try:
            logger.info("Reading configuration YAML file: {config_filepath}")
            self.config = read_yaml(config_filepath)

            logger.info("Reading schema YAML file: {schema_filepath}")
            self.schema = read_yaml(schema_filepath)

            logger.info("Reading parameters YAML file: {params_filepath}")
            self.params = read_yaml(params_filepath)

            # Ensure the artifacts root directory is created
            logger.info(f"Creating artifacts root directory: {self.config.artifacts_root}")
            create_directories([self.config.artifacts_root], True)

        except Exception as e:
            logger.error("Error initializing ConfigurationManager.", exc_info=True)
            raise CustomException(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Returns the DataIngestionConfig object with settings for the data 
        ingestion stage, ensuring necessary directories are created.
        """
        try:
            logger.info("Fetching data ingestion configuration.")
            config = self.config.data_ingestion

            # Accessing the parameters from the parameters YAML file.
            directory_names = self.params.directory_names

            # Ensure the data ingestion root directory is created
            logger.info(f"Creating data ingestion root directory: {config.root_dir}")
            create_directories([config.root_dir], True)

            # Create and return DataIngestionConfig object
            data_ingestion_config = DataIngestionConfig(
                root_dir=config.root_dir,
                dataset_identifier=config.dataset_identifier,
                input_data_folder = directory_names.input_data_folder,
                new_dir_name = directory_names.new_dir_name
            )
            logger.info("Data ingestion configuration created successfully.")
            return data_ingestion_config
        except Exception as e:
            logger.error("Error while fetching data ingestion configuration: {e}", exc_info=True)
            raise CustomException(e, sys)


    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        try:
            config = self.config.data_preprocessing
            image_preprocessing = self.params.image_preprocessing

            # Ensure the root directory for data preprocessing artifacts exists
            logger.info(f"Creating data preprocessing root directory: {config.root_dir}")
            create_directories([config.root_dir])

            # Create a DataTransformationConfig object with the retrieved configurations
            data_preprocessing_config = DataPreprocessingConfig(
                    root_dir = config.root_dir,
                    raw_train = config.raw_train,
                    raw_valid = config.raw_valid,
                    raw_test = config.raw_test,
                    resize_train = config.resize_train,
                    resize_valid = config.resize_valid,
                    resize_test = config.resize_test,
                    preprocessor_path = config.preprocessor_path,
                    target_size = tuple(image_preprocessing.target_size)
                )

            logger.info("Data preprocessing configuration successfully created.")
            return data_preprocessing_config
    
        except Exception as e:
            # Log the error and raise a custom exception
            logger.error(f"Error while fetching data transformation configuration: {e}", exc_info=True)
            raise CustomException(e, sys) 


    def get_dataset_loader_config(self) -> DatasetLoaderConfig:
        try:
            config = self.config.dataset_loader
            image_preprocessing = self.params.image_preprocessing

            # Ensure the root directory for dataset loader artifacts exists
            logger.info(f"Creating dataset loader root directory: {config.root_dir}")
            create_directories([config.root_dir])

            # Create a DatasetLoaderConfig object with the retrieved configurations
            dataset_loader_config = DatasetLoaderConfig(
                    root_dir = config.root_dir,
                    resize_train = config.resize_train,
                    resize_valid = config.resize_valid,
                    resize_test = config.resize_test,
                    preprocessor_path = config.preprocessor_path,
                    class_labels_path = config.class_labels_path,
                    train_ds_path = config.train_ds_path,
                    valid_ds_path = config.valid_ds_path,
                    test_ds_path = config.test_ds_path,
                    target_size = tuple(image_preprocessing.target_size),
                    batch_size = image_preprocessing.batch_size,
                    shuffle = image_preprocessing.shuffle
                )

            logger.info("Dataset Loader configuration successfully created.")
            return dataset_loader_config
    
        except Exception as e:
            # Log the error and raise a custom exception
            logger.error(f"Error while fetching dataset loader configuration: {e}", exc_info=True)
            raise CustomException(e, sys) 

    def model_builder_config(self) -> ModelBuilderConfig:
        try:
            config = self.config.model_builder
            model_builder_params = self.params.model_builder
            common_params = self.params.common_params

            # Ensure the root directory for Model Builder artifacts exists
            logger.info(f"Creating Model Builder root directory: {config.root_dir}")
            create_directories([config.root_dir])

            # Create a ModelBuilderConfig object with the retrieved configurations
            model_builder_config = ModelBuilderConfig(
                    root_dir = config.root_dir,
                    model_summaries = config.model_summaries,
                    model_names = model_builder_params.model_names,
                    input_shape = tuple(common_params.input_shape),
                    include_top = model_builder_params.include_top,
                    freeze_all = model_builder_params.freeze_all,
                    num_classes = common_params.num_classes,
                    learning_rate = model_builder_params.learning_rate,
                    loss_function = model_builder_params.loss_function,
                    optimizer = model_builder_params.optimizer,
                    freeze_till = model_builder_params.freeze_till,
                )

            logger.info("Model Builder configuration successfully created.")
            return model_builder_config
    
        except Exception as e:
            # Log the error and raise a custom exception
            logger.error(f"Error while fetching model builder configuration: {e}", exc_info=True)
            raise CustomException(e, sys) 


    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        """Prepare and return the callback configuration for model training."""
        try:
            # Fetch callback configuration from the config object
            config = self.config.prepare_callbacks

            # Create the root directory for Model Builder artifacts
            logger.info(f"Creating root directory for Model Builder artifacts: {config.root_dir}")
            create_directories([config.root_dir])

            # Create directory for TensorBoard logs
            logger.info(f"Creating TensorBoard log directory: {config.tensorboard_log_dir}")
            create_directories([config.tensorboard_log_dir])

            # Create directory for model checkpoint files
            model_ckpt_dir = os.path.dirname(config.checkpoint_filepath)
            logger.info(f"Creating model checkpoint directory: {model_ckpt_dir}")
            create_directories([Path(model_ckpt_dir)])

            # Prepare and return the callback configuration
            prepare_callback_config = PrepareCallbacksConfig(
                root_dir=config.root_dir,
                tensorboard_log_dir=config.tensorboard_log_dir,
                checkpoint_filepath=config.checkpoint_filepath
            )
            logger.info("Callback configuration prepared successfully.")
            return prepare_callback_config

        except Exception as e:
            # Log and raise a custom exception if an error occurs
            logger.error("Error occurred while preparing callback configuration.", exc_info=True)
            raise CustomException(e, sys)


    def get_model_trainer_evaluation_config(self) -> ModelTrainerEvaluationConfig:
        try:
            # Accessing the model trainer configuration from the main configuration YAML.
            config = self.config.model_trainer_evaluation   
            
            # Accessing the parameters from the parameters YAML file.
            model_trainer_params = self.params.model_trainer
            mlflow_params = self.params.mlflow

            # Ensuring the root directory for the model trainer is created.
            create_directories([config.root_dir])

            # Creating and returning a ModelTrainerConfig object with the required configurations.
            model_trainer_evaluation_config = ModelTrainerEvaluationConfig(
                root_dir = config.root_dir,
                train_ds = config.train_ds,
                valid_ds = config.valid_ds,
                class_labels_path = config.class_labels_path,
                best_base_model_path = config.best_base_model_path,
                base_metrics_file = config.base_metrics_file,
                base_cm_plot = config.base_cm_plot,
                epochs = model_trainer_params.epochs,
                mlflow_experiment_name = mlflow_params.experiment_name
            )

            return model_trainer_evaluation_config

        except Exception as e:
            # Logging the error and raising a custom exception if fetching configuration fails.
            logger.error("Error fetching model trainer evaluation configuration.", exc_info=True)
            raise CustomException(e, sys)



    def get_hyperpameter_tuning_config(self) -> HyperparameterTuningConfig:
        try:
            # Accessing the hyperparameter tuning configuration from the main configuration YAML.
            config = self.config.hyperparameter_tuning   
            
            # Accessing the parameters from the parameters YAML file.
            hyper_tuning_params = self.params.hyper_tuning
            common_params = self.params.common_params
            mlflow_params = self.params.mlflow

            # Ensuring the root directory for the model trainer is created.
            create_directories([config.root_dir])

            # Creating and returning a ModelTrainerConfig object with the required configurations.
            hyperpameter_tuning_config = HyperparameterTuningConfig(
                root_dir = config.root_dir,
                train_ds = config.train_ds,
                valid_ds = config.valid_ds,
                test_ds = config.test_ds,
                class_labels_path = config.class_labels_path,
                best_base_model_path = config.best_base_model_path,
                tuned_model_metrics_file = config.tuned_model_metrics_file,
                tuned_model_cm_plot = config.tuned_model_cm_plot,
                tuned_model_path = config.tuned_model_path,
                input_shape = tuple(common_params.input_shape),
                num_classes = common_params.num_classes,
                max_trials = hyper_tuning_params.max_trials,
                epochs = hyper_tuning_params.epochs,
                mlflow_experiment_name = mlflow_params.experiment_name
            )

            return hyperpameter_tuning_config

        except Exception as e:
            # Logging the error and raising a custom exception if fetching configuration fails.
            logger.error("Error fetching hyperparameter tuning configuration.", exc_info=True)
            raise CustomException(e, sys)