from dataclasses import dataclass  # Importing the dataclass decorator to simplify data class creation.
from pathlib import Path  # Pathlib module for working with filesystem paths in an object-oriented way.
from typing import List, Tuple, Optional
#----------------DataIngestionConfig------------------------
# Define a data class to store configuration settings for data ingestion.
@dataclass
class DataIngestionConfig:
    """
    A configuration class for managing settings related to data ingestion.
    This class leverages the @dataclass decorator to automatically generate 
    initialization, representation, and comparison methods.
    Note:
        This class is often used as the return type from a configuration setup 
        function. Using it as a return type ensures that all configuration 
        properties are encapsulated in a single object, promoting type safety, 
        readability, and ease of use across different parts of the codebase.
    """

    root_dir: Path  # Path to the root directory for data ingestion operations.
    dataset_identifier: str # URL of the source data.
    input_data_folder: str # Name of the folder where the dataset will be downloaded and extracted
    new_dir_name: str #standardized name for the dataset directory after extraction and renaming

#----------------DataPreprocessingConfig------------------------
# Define a data class to store configuration settings for data preprocessing.
@dataclass
class DataPreprocessingConfig:
    root_dir: Path                  #Root directory for preprocessing-related artifacts
    raw_train: Path                 #Path to the raw train input data
    raw_valid: Path                 #Path to the raw validation input data
    raw_test: Path                  #Path to the raw test input data
    resize_train: Path              #Path to the resized train input data to be saved
    resize_valid: Path              #Path to the resized validation input data to be saved
    resize_test: Path               #Path to the resized test input data to be saved
    preprocessor_path: Path         #Path to the preprocessor object file to be saved
    target_size: Tuple[int, int]    #Target size of the images      


#----------------DataLoaderConfig------------------------
# Define a data class to store configuration settings for data loader.
@dataclass
class DatasetLoaderConfig:
    root_dir: Path                  # Root directory for preprocessing-related artifacts
    resize_train: Path              # Path to the resized train input data
    resize_valid: Path              # Path to the resized validation input data
    resize_test: Path               # Path to the resized test input data
    preprocessor_path: Path         # Path to the preprocessor object file
    class_labels_path: Path         # Path to save class labels
    train_ds_path: Path             # Path to save the train dataset
    valid_ds_path: Path             # Path to save the validation dataset
    test_ds_path: Path              # Path to save the test dataset
    target_size: Tuple[int, int]    # Target size of the images (height, width)
    batch_size: int                 # Batch size (fixed typo)
    shuffle: bool                   # Whether to shuffle the dataset


#----------------ModelBuilderConfig------------------------
# Define a data class to store configuration settings for Model Builder.
@dataclass
class ModelBuilderConfig:
    root_dir: Path                      # Root directory for storing models and artifacts
    model_summaries: Path
    model_names: List[str]              # List of model names to be used (e.g., VGG16, ResNet50, etc.)
    input_shape: Tuple[int, int, int]   # Input shape of images (height, width, channels)
    include_top: bool                   # Whether to include the top (fully connected) layers of pre-trained models
    freeze_all: bool                    # If True, freezes all layers in the pre-trained model
    num_classes: int                    # Number of output classes for classification
    learning_rate: float                # Learning rate for training
    loss_function: str                  # Loss function to use (e.g., "sparse_categorical_crossentropy")
    optimizer: str                      # Optimizer type (e.g., "adam", "sgd")
    freeze_till: Optional[int] = None   # Number of layers to keep frozen (None = all frozen)

@dataclass
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_log_dir: Path
    checkpoint_filepath: Path


#-------------Model Trainer---------------------------------------
# Define a data class to store configuration settings for model trainer
@dataclass
class ModelTrainerEvaluationConfig:
    root_dir: Path                  # Path to the root directory for training-related artifacts
    train_ds: Path      
    valid_ds: Path
    class_labels_path: Path
    best_base_model_path: Path
    base_metrics_file: Path
    base_cm_plot: Path
    epochs: int
    mlflow_experiment_name: str
    

# Define a data class to store configuration settings for model trainer
@dataclass
class HyperparameterTuningConfig:
    root_dir: Path
    train_ds: Path
    valid_ds: Path
    test_ds: Path
    class_labels_path: Path
    best_base_model_path: Path
    tuned_model_metrics_file: Path
    tuned_model_cm_plot: Path
    tuned_model_path: Path
    input_shape: Tuple[int, int, int]   # Input shape of images (height, width, channels)    
    num_classes: int                    # Number of output classes for classification
    max_trials: int
    epochs: int
    mlflow_experiment_name: str     # Experiment name for for the MLflow tracking server where metrics and models will be logged.
                                    # to be logged into MLflow for tracking.