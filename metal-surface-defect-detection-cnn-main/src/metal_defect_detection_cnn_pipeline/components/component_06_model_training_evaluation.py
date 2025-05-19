import numpy as np
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tensorflow as tf

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

from src.metal_defect_detection_cnn_pipeline.entity.config_entity import ModelTrainerEvaluationConfig

import mlflow
import mlflow.sklearn
from src.metal_defect_detection_cnn_pipeline.constants.constant_values import (
    mlflow_uri,
    username,
    password
)

#Set MLflow tracking URI and authentication details
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = password
mlflow.set_registry_uri(mlflow_uri)
logger.info("MLflow tracking URI and authentication set.")


if mlflow.active_run():
    mlflow.end_run()  

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


class ModelTrainerEvaluation:
    """
    A class to train, evaluate, and fine-tune machine learning models on different resampled datasets.
    It includes steps like cross-validation, training and evaluating base models, identifying the best
    performing model, and tuning hyperparameters for improved performance.
    """

    def __init__(self, config: ModelTrainerEvaluationConfig):
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
            self.results = {}

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

    def train_and_evaluate(self):
        """
        Train all models in self.models_dict, evaluate their performance on the validation dataset,
        and store their metrics in self.results.
        """
        for model_name, model in self.models_dict.items():
            try:
                logger.info(f"Training {model_name}...")

                # Create a unique callback instance for each model
                callbacks = PrepareCallback(
                    PrepareCallbacksConfig(
                        root_dir=prepare_callbacks_config.root_dir,
                        tensorboard_log_dir=os.path.join(
                            prepare_callbacks_config.tensorboard_log_dir, f"base_{model_name}"
                        ),
                        checkpoint_filepath=os.path.join(
                            os.path.dirname(prepare_callbacks_config.checkpoint_filepath),
                            f"base_{model_name}.h5"
                        ),
                    )
                ).get_callbacks()

                # Train the model and capture training history
                history = model.fit(
                    self.train_ds,
                    validation_data=self.valid_ds,
                    epochs=self.config.epochs,
                    verbose=1,
                    callbacks=callbacks
                )

                # Extract metrics from training history
                train_acc = history.history["accuracy"][-1]  # Last epoch training accuracy
                val_acc = history.history["val_accuracy"][-1]  # Last epoch validation accuracy
                val_loss = history.history["val_loss"][-1]  # Last epoch validation loss

                # Extract true and predicted labels from the validation dataset
                y_true, y_pred = self._get_true_and_pred_labels(model, self.valid_ds)

                # Calculate performance metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average="weighted")
                recall = recall_score(y_true, y_pred, average="weighted")
                f1 = f1_score(y_true, y_pred, average="weighted")
                conf_matrix = confusion_matrix(y_true, y_pred)
                class_report = classification_report(y_true, y_pred, target_names=self.class_names)

                # Store evaluation results
                self.results[model_name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "classification_report": class_report,
                    "train_accuracy": train_acc,
                    "validation_loss": val_loss,
                    "validation_accuracy": val_acc
                }

                # Log results to MLflow
                with mlflow.start_run(run_name=model_name, nested=True):
                    mlflow.log_params({"model_name": model_name})

                    # Log individual metrics
                    mlflow.log_metrics({k: v for k, v in self.results[model_name].items() if isinstance(v, (int, float))})

                    # Log classification report
                    mlflow.log_text(self.results[model_name]['classification_report'], "classification_report.txt")

                    # Save confusion matrix as a text file and Log confusion matrix text
                    conf_matrix_str = np.array2string(conf_matrix, separator=", ")
                    mlflow.log_text(conf_matrix_str, "confusion_matrix.txt")
                    print(f"Metrics logged to MLflow for model: {model_name}")


                # Display and log metrics
                logger.info(f"Model: {model_name}, Accuracy: {accuracy:.4f}")
                print(f"\nModel: {model_name}")
                print(f"Accuracy: {accuracy:.4f}")
                print("Classification Report:\n", class_report)

                # Plot and save the confusion matrix
                self._plot_confusion_matrix(conf_matrix, model_name)

            except Exception as e:
                logger.error(f"Error while training or evaluating {model_name}: {str(e)}")
                raise CustomException(e, sys)

        # Save the metrics to a JSON file
        try:
            save_json(Path(self.config.base_metrics_file), self.results)
            logger.info(f"Training and evaluation results saved to {self.config.base_metrics_file}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")
            raise CustomException(e, sys)

    def _get_true_and_pred_labels(self, model, dataset):
        """Extract true and predicted labels from a TensorFlow dataset."""
        y_true, y_pred = [], []
        try:
            for batch in dataset:
                X_batch, y_batch = batch
                y_true.extend(y_batch.numpy().astype(int).flatten())
                preds = model.predict(X_batch)
                y_pred.extend(np.argmax(preds, axis=1))
        except Exception as e:
            logger.error(f"Error while extracting true and predicted labels: {str(e)}")
            raise CustomException(e, sys)
        return np.array(y_true), np.array(y_pred)

    def _plot_confusion_matrix(self, conf_matrix, model_name):
        """Plot and save the confusion matrix as a heatmap."""
        try:
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.class_names,
                yticklabels=self.class_names
            )
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title(f"Confusion Matrix - {model_name}")

            # Save the confusion matrix plot
            filename = f"base_model_{model_name.lower()}_valid_ds_confusion_matrix.jpg"
            save_path = os.path.join(self.config.base_cm_plot, filename)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
            logger.info(f"Confusion matrix saved as {save_path}")

        except Exception as e:
            logger.error(f"Error while plotting confusion matrix for {model_name}: {str(e)}")
            raise CustomException(e, sys)

    def get_best_model(self):
        """
        Select the best-performing model based on a weighted scoring system.

        The scoring formula considers:
        - Validation Accuracy: Higher accuracy is better.
        - Validation Loss: Lower loss indicates better generalization.
        - Overfitting Penalty: Models with a large gap between training and validation accuracy are penalized.

        The model with the highest adjusted score is selected.
        """
        try:
            def score_model(model_name, result):
                val_acc = result["accuracy"]
                val_loss = result["validation_loss"]
                train_acc = result["train_accuracy"]

                # Overfitting penalty: Absolute difference between training and validation accuracy
                overfit_penalty = abs(train_acc - val_acc)

                # Weighted scoring formula
                score = val_acc - overfit_penalty - (val_loss * 0.1)
                return score

            # Find the best-performing model
            best_model_name = max(self.results, key=lambda k: score_model(k, self.results[k]))

            # Save the best model name to a JSON file
            with open(self.config.best_base_model_path, "w") as f:
                json.dump({"best_base_model": best_model_name}, f, indent=4)

            logger.info(
                f"Best Performing Base Model: {best_model_name} with Adjusted Score: "
                f"{score_model(best_model_name, self.results[best_model_name]):.4f}"
            )
            return best_model_name, self.models_dict[best_model_name]

        except Exception as e:
            logger.error(f"Error while selecting the best model: {str(e)}")
            raise CustomException(e, sys)

    def run_train_evaluate(self):
        """Train and evaluate models, then get the best model."""
        try:
            self.train_and_evaluate()
            best_model_name, best_model = self.get_best_model()
            logger.info(f"Best model '{best_model_name}' training and evaluation completed successfully.")
        except Exception as e:
            logger.error(f"Error in run_train_evaluate: {str(e)}")
            raise CustomException(e, sys)