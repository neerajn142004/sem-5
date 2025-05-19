import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tempfile

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

from tensorflow.keras import layers, models, applications, Model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.metal_defect_detection_cnn_pipeline.utils.custom_logging import logger
from src.metal_defect_detection_cnn_pipeline.utils.custom_exception import CustomException
from src.metal_defect_detection_cnn_pipeline.utils.common import (
    save_bin,
    save_json
)

from src.metal_defect_detection_cnn_pipeline.entity.config_entity import HyperparameterTuningConfig

import mlflow
import mlflow.sklearn

from src.metal_defect_detection_cnn_pipeline.constants.constant_values import (
    mlflow_uri,
    username,
    password
)

# Set MLflow tracking URI and authentication details
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = password
mlflow.set_registry_uri(mlflow_uri)
logger.info("MLflow tracking URI and authentication set.")

# Determine the artifact storage type
tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

#if mlflow.active_run():
#    mlflow.end_run()  




# Import ModelBuilderPipeline to build models and return them as a dictionary
from src.metal_defect_detection_cnn_pipeline.pipeline.stage_04_model_builder import ModelBuilderPipeline

# Create an instance of ModelBuilderPipeline
model_builder = ModelBuilderPipeline()

# Call the main() method to generate models and store them in a dictionary
models_dict = model_builder.main()

# Import PrepareCallbacksConfig and PrepareCallback for setting callbacks
from src.metal_defect_detection_cnn_pipeline.entity.config_entity import PrepareCallbacksConfig
from src.metal_defect_detection_cnn_pipeline.components.component_05_prepare_callbacks import PrepareCallback

# Import ConfigurationManager to retrieve configuration settings
from src.metal_defect_detection_cnn_pipeline.config.configuration import ConfigurationManager

# Create an instance of ConfigurationManager to manage configuration settings
config_manager = ConfigurationManager()

# Retrieve PrepareCallbacksConfig object containing callback configurations
prepare_callbacks_config = config_manager.get_prepare_callback_config()


class ModelTuner:
    """
    ModelTuner class for fine-tuning an image classification model using Keras Tuner.

    - Initializes with a dictionary of models and a selected base model.
    - Uses Bayesian Optimization to tune hyperparameters.
    - Evaluates the best model on validation and test datasets.
    - Saves evaluation results and plots confusion matrices.
    """
    def __init__(self, config: HyperparameterTuningConfig):
        """
        Initializes the ModelTrainerEvaluation with the given configuration.

        Args:
            config (ModelTrainerEvaluationConfig): Configuration object containing paths and parameters
        """
        self.config = config
        mlflow.set_experiment(self.config.mlflow_experiment_name)  # Set experiment name for MLflow
        # Log the configuration details at initialization
        logger.info("Initializing ModelTrainerEvaluation with the following configuration:")
        for key, value in vars(self.config).items():
            logger.info(f"{key}: {value}")

        try:
            # Load models and class labels
            self.models_dict = models_dict
            self.class_names = self.load_class_labels()

            # Load training and validation datasets
            self.train_ds = tf.data.Dataset.load(self.config.train_ds)
            self.valid_ds = tf.data.Dataset.load(self.config.valid_ds)
            self.test_ds = tf.data.Dataset.load(self.config.test_ds)

            self.tuner = None  # Keras Tuner instance (initialized later)
            self.best_base_model_name, self.base_model = self._get_base_model()  # Retrieve and freeze base model

        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise CustomException(e, sys)   

    def load_class_labels(self):
        """Load class labels from a JSON file."""
        try:
            with open(self.config.class_labels_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error decoding class labels JSON file: {str(e)}")
            raise CustomException(e, sys)

    def _get_base_model(self):

        with open(self.config.best_base_model_path, "r") as f:
            data = json.load(f)
            best_base_model_name = data["best_base_model"]
            print(best_base_model_name)

        if best_base_model_name not in self.models_dict:
            raise ValueError(f"Invalid base model: {best_base_model_name}")

        base_model = self.models_dict[best_base_model_name]
        
        if isinstance(base_model, keras.Sequential):  # Custom CNN model
            return best_base_model_name, base_model
        elif isinstance(base_model, Model):  # Pretrained model
            base_model.trainable = False  # Freeze the model
            return best_base_model_name, base_model
        else:
            raise TypeError("Unsupported model type. Must be Sequential or Model.")


    def build_tuned_model(self, hp):
        """Builds a model with tunable hyperparameters."""
        inputs = layers.Input(shape=self.config.input_shape)
        x = self.base_model(inputs, training=False)  # Ensure compatibility with functional API
        
        if len(x.shape) == 2:  # If output is a flattened vector, skip pooling
            x = layers.Dense(hp.Int("dense_units", min_value=128, max_value=512, step=64), activation="relu")(x)
        else:
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(hp.Int("dense_units", min_value=128, max_value=512, step=64), activation="relu")(x)
        
        x = layers.Dropout(hp.Float("dropout_rate", min_value=0.2, max_value=0.6, step=0.1))(x)
        outputs = layers.Dense(self.config.num_classes, activation="softmax")(x)

        model = Model(inputs, outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def tune_model(self):
        """Tunes the model using Bayesian Optimization."""
        model_name = f"tuned_{self.best_base_model_name}"
        # Create a unique callback instance for each model
        callbacks = PrepareCallback(
            PrepareCallbacksConfig(
                root_dir=prepare_callbacks_config.root_dir,
                tensorboard_log_dir=os.path.join(
                    prepare_callbacks_config.tensorboard_log_dir, f"{model_name}"
                ),
                checkpoint_filepath=os.path.join(
                    os.path.dirname(prepare_callbacks_config.checkpoint_filepath),
                    f"{model_name}.h5"
                ),
            )
        ).get_callbacks()


        temp_dir = tempfile.mkdtemp()  # Temporary directory for tuner results
        self.tuner = kt.BayesianOptimization(
            self.build_tuned_model,
            objective="val_accuracy",
            max_trials=self.config.max_trials,
            num_initial_points=3,
            directory=temp_dir,  # Store results in a temporary directory
            project_name=None
            #directory="tuner_results",
            #project_name="image_classification_bayesian"            
        )
        
        self.tuner.search(self.train_ds, validation_data=self.valid_ds, epochs=self.config.epochs, callbacks=callbacks)
        
        # Retrieve and print best hyperparameters
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Best hyperparameters: Dense units = {best_hps.get('dense_units')}, Dropout rate = {best_hps.get('dropout_rate')}")

    def evaluate_model(self):
        """Evaluates the best model and saves results."""

        best_model = self.tuner.get_best_models(num_models=1)[0]
        
        def evaluate_dataset(dataset, dataset_name):
            """Evaluates model on a dataset and returns performance metrics."""
            y_true, y_pred = [], []
            
            for images, labels in dataset:
                preds = best_model.predict(images)
                y_true.extend(labels.numpy())
                y_pred.extend(np.argmax(preds, axis=1))
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            conf_matrix = confusion_matrix(y_true, y_pred)
            class_report = classification_report(y_true, y_pred, target_names=self.class_names)
            
            # Plot and save confusion matrix
            self._plot_confusion_matrix(conf_matrix, self.class_names, dataset_name)

            # Convert confusion matrix to string format
            conf_matrix_str = np.array2string(conf_matrix, separator=", ")


            # Log results to MLflow
            with mlflow.start_run(run_name=f"tuned_{self.best_base_model_name}_{dataset_name}_data", nested=True):
                mlflow.log_params({"model_name": self.best_base_model_name, "dataset": dataset_name})

                # Log individual metrics
                mlflow.log_metrics(
                    {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                    }
                )

                # Log classification report as a text file
                mlflow.log_text(class_report, f"{dataset_name}_classification_report.txt")

                # Log confusion matrix as a text file
                mlflow.log_text(conf_matrix_str, f"{dataset_name}_confusion_matrix.txt")

                print(f"Metrics logged to MLflow for model: {self.best_base_model_name} on {dataset_name} dataset.")

            return {
                "dataset": dataset_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "classification_report": class_report,
            }
        
        validation_results = evaluate_dataset(self.valid_ds, "Validation")
        test_results = evaluate_dataset(self.test_ds, "Test")
        
        evaluation_results = {"validation": validation_results, "test": test_results}
        
        save_json(Path(self.config.tuned_model_metrics_file), evaluation_results)
        print(f"Evaluation results saved to {self.config.tuned_model_metrics_file}")

        best_model.save(self.config.tuned_model_path)
        print(f"Tuned model saved to {self.config.tuned_model_path}")

        with mlflow.start_run(run_name=f"tuned_{self.best_base_model_name}_model_registration", nested=True):
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    best_model,
                    "model",
                    registered_model_name=f"tuned_model_{self.best_base_model_name}",
                )
                print(f"Model registered to MLflow as 'tuned_model_{self.best_base_model_name}'.")
            else:
                mlflow.sklearn.log_model(best_model, "model")
                print("Model logged to MLflow as a local artifact.")
     
        return evaluation_results

    def _plot_confusion_matrix(self, conf_matrix, class_labels, dataset_name):
        """Plots and saves the confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix - {dataset_name} Dataset")
        filename = f"tuned_model_{dataset_name.lower()}_confusion_matrix.jpg"
        save_path = os.path.join(self.config.tuned_model_cm_plot, filename)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"Confusion matrix saved as {save_path}")
       

    def initiate_hyper_tuning(self):
        """Initiates the hyperparameter tuning and evaluation process."""
        try:
            logger.info("Initiating hyperparameter tuning...")            
            # Perform model tuning
            self.tune_model()
            
            # Evaluate the best model and save results
            self.evaluate_model()
            logger.info("Hyperparameter tuning and model evaluation completed successfully.")

        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {str(e)}")
            raise CustomException(e, sys)        