import sys
import json
import numpy as np
import tensorflow as tf

from pathlib import Path
from PIL import Image
from src.metal_defect_detection_cnn_pipeline.utils.common import load_bin
from src.metal_defect_detection_cnn_pipeline.utils.custom_logging import logger
from src.metal_defect_detection_cnn_pipeline.utils.custom_exception import CustomException

# Define paths for model, preprocessor, and class labels
model_path = Path("artifacts/af_07_hyperparameter_tuning/tuned_model.h5")
preprocessor_path = Path("artifacts/af_02_data_preprocessing/pre_processor.joblib")
class_labels_path = Path("artifacts/af_03_dataset_loader/class_labels.json")


class PredictionPipeline:
    def __init__(self):
        """
        Initializes the pipeline by loading the necessary objects such as the preprocessor,
        class labels, and the tuned model. Handles errors during the loading process.
        """
        try:
            # Load the trained model
            logger.info(f"Loading tuned model from: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            logger.info("Tuned model loaded successfully.")

            # Load preprocessor object
            logger.info(f"Loading preprocessor from: {preprocessor_path}")
            if not preprocessor_path.exists():
                raise FileNotFoundError(f"File not found: {preprocessor_path}")

            self.preprocessor = load_bin(preprocessor_path)
            logger.info("Preprocessor loaded successfully.")

            # Load class labels
            logger.info(f"Loading class labels from: {class_labels_path}")
            with open(class_labels_path, "r") as f:
                self.class_labels = json.load(f)
            logger.info("Class labels loaded successfully.")

        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise CustomException(e, sys)

    def _preprocess_image(self, image_path):
        """
        Preprocesses the input image before making predictions.

        Args:
            image_path (str): Path to the input image.

        Returns:
            np.array: Preprocessed image ready for prediction.
        """
        try:
            # Load and preprocess the input image
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = img.resize(self.preprocessor.target_size, Image.Resampling.LANCZOS)

            img_array = np.array(img) / 255.0  # Normalize image to [0, 1]

            # Apply normalization if mean and std are available
            if self.preprocessor.train_mean is not None and self.preprocessor.train_std is not None:
                img_array = self.preprocessor.normalize_image(img_array)

            # Add batch dimension for prediction
            img_array = np.expand_dims(img_array, axis=0)
            logger.info("Image preprocessed successfully.")
            return img_array

        except Exception as e:
            logger.error(f"Error during image preprocessing: {str(e)}")
            raise CustomException(e, sys)

    def predict(self, image_path):
        """
        Predicts the class of the input image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            str: Predicted class label with confidence score.
        """
        try:
            # Preprocess the image
            logger.info(f"Preprocessing image: {image_path}")
            img_array = self._preprocess_image(image_path)

            # Make predictions
            logger.info("Making predictions with the tuned model...")
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions)

            # Get predicted class label
            predicted_label = self.class_labels[predicted_class]
            logger.info(f"Prediction: {predicted_label} with confidence {confidence:.2f}")

            # Return result
            return f"Predicted Class: {predicted_label} (Confidence: {confidence:.2f})"

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise CustomException(e, sys)