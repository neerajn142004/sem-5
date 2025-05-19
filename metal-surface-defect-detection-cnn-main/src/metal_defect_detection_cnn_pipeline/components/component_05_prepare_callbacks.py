import os
import sys
import shutil
from time import strftime

# Disable oneDNN optimizations for TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

from src.metal_defect_detection_cnn_pipeline.utils.custom_logging import logger
from src.metal_defect_detection_cnn_pipeline.utils.custom_exception import CustomException
from src.metal_defect_detection_cnn_pipeline.entity.config_entity import PrepareCallbacksConfig


class PrepareCallback:
    """
    A class to create and manage TensorFlow training callbacks.

    This class provides:
    - TensorBoard logging for visualization.
    - ModelCheckpoint to save the best model.
    - EarlyStopping to stop training when validation loss stops improving.
    - ReduceLROnPlateau to reduce the learning rate when validation performance plateaus.

    Attributes:
        tensorboard_log_dir (str): Directory to save TensorBoard logs.
        checkpoint_filepath (str): File path to save the best model checkpoint.
        patience (int): Number of epochs to wait before stopping training (default: 5).
        factor (float): Factor to reduce the learning rate when performance plateaus (default: 0.1).
    """

    def __init__(self, config: PrepareCallbacksConfig):
        """
        Initializes the PrepareCallback class with the provided configuration.
        """
        try:
            self.config = config
            logger.info("Initializing PrepareCallback with the following configuration:")
            for key, value in vars(self.config).items():
                logger.info(f"{key}: {value}")

            self.patience = 5
            self.factor = 0.1

        except Exception as e:
            logger.error(f"Error initializing PrepareCallback: {e}")
            raise CustomException(e, sys)

    def clear_tensorboard_logs(self):
        """Deletes old TensorBoard logs before creating a new run."""
        if os.path.exists(self.config.tensorboard_log_dir):
            shutil.rmtree(self.config.tensorboard_log_dir)  # Remove old logs
        os.makedirs(self.config.tensorboard_log_dir, exist_ok=True)  # Create fresh log directory

    def create_tensorboard_callback(self):
        """Clears old logs and creates a TensorBoard callback with a new log directory."""
        self.clear_tensorboard_logs()  # Remove previous logs
        timestamp = strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(self.config.tensorboard_log_dir, f"tb_logs_at_{timestamp}")
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    def create_checkpoint_callback(self):
        """Creates a ModelCheckpoint callback to save the best model during training."""
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.checkpoint_filepath,
            save_best_only=True
        )

    def create_early_stopping_callback(self):
        """Creates an EarlyStopping callback to stop training if validation loss doesn't improve."""
        return tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True
        )

    def create_reduce_lr_callback(self):
        """Creates a ReduceLROnPlateau callback to reduce learning rate when validation loss stops improving."""
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=self.factor,
            patience=self.patience // 2,
            verbose=1
        )

    def get_callbacks(self):
        """Returns a list of all defined callbacks."""
        return [
            self.create_tensorboard_callback(),
            self.create_checkpoint_callback(),
            self.create_early_stopping_callback(),
            self.create_reduce_lr_callback()
        ]