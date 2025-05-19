import os
import sys
import io
from pathlib import Path
# Disable oneDNN optimizations for TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, models


from src.metal_defect_detection_cnn_pipeline.utils.custom_logging import logger
from src.metal_defect_detection_cnn_pipeline.utils.custom_exception import CustomException
from src.metal_defect_detection_cnn_pipeline.entity.config_entity import ModelBuilderConfig

class ModelBuilder:
    """
    A utility class for loading image datasets, normalizing images using precomputed statistics,
    and extracting class labels.

    Features:
    - Loads image datasets from a specified directory.
    - Normalizes images using precomputed mean and standard deviation.
    - Extracts class names and saves them to a JSON file.
    - Supports batch loading and resizing.
    """

    def __init__(self, config: ModelBuilderConfig):
        """
        Initializes the DatasetLoader class with the provided configuration.
        """
        try:
            self.config = config
            logger.info("Initializing ModelBuilderConfig with the following configuration:")
            for key, value in vars(self.config).items():
                logger.info(f"{key}: {value}")

            self.models_dict = {}  # Dictionary to store models
            self._validate_config()

        except Exception as e:
            logger.error(f"Error initializing ModelBuilder: {e}")
            raise CustomException(e, sys)
        
    
    def _validate_config(self):
        """Validate model configuration."""
        if not isinstance(self.config.model_names, list) or not self.config.model_names:
            raise ValueError("model_names should be a non-empty list of model names.")

    def _load_pretrained_model(self, model_name):
        """Load a pretrained model based on model_name."""
        try:
            model_name = model_name.lower()
            
            if model_name == "vgg16":
                base_model = tf.keras.applications.VGG16(input_shape=self.config.input_shape, 
                                                        weights="imagenet", 
                                                        include_top=self.config.include_top)
            elif model_name == "resnet50":
                base_model = tf.keras.applications.ResNet50(input_shape=self.config.input_shape, 
                                                            weights="imagenet", 
                                                            include_top=self.config.include_top)
            elif model_name == "mobilenetv2":
                base_model = tf.keras.applications.MobileNetV2(input_shape=self.config.input_shape, 
                                                            weights="imagenet", 
                                                            include_top=self.config.include_top)
            elif model_name == "efficientnetb0":
                base_model = tf.keras.applications.EfficientNetB0(input_shape=self.config.input_shape, 
                                                                weights="imagenet", 
                                                                include_top=self.config.include_top)
            else:
                raise ValueError(f"Unsupported model name: {model_name}")

            return self._freeze_layers(base_model)
        
        except Exception as e:
            logger.error(f"Error loading pretrained model {model_name}: {e}")
            raise CustomException(e, sys)        

    def _freeze_layers(self, model):
        """Freeze layers based on configuration for transfer learning."""
        try:
            if self.config.freeze_all:
                for layer in model.layers:
                    layer.trainable = False  # Freeze all layers
            elif self.config.freeze_till is not None and self.config.freeze_till > 0:
                for layer in model.layers[:-self.config.freeze_till]:
                    layer.trainable = False  # Freeze some layers
            return model
        except Exception as e:
            logger.error(f"Error freezing layers: {e}")
            raise CustomException(e, sys)

    def _add_classification_head(self, base_model):
        """Attach a classification head to the base model."""
        try:
            flatten = layers.Flatten()(base_model.output)
            dense = layers.Dense(256, activation="relu")(flatten)
            dropout = layers.Dropout(0.5)(dense)
            output = layers.Dense(self.config.num_classes, activation="softmax")(dropout)

            model = tf.keras.Model(inputs=base_model.input, outputs=output)
            return self._compile_model(model)
        except Exception as e:
            logger.error(f"Error adding classification head: {e}")
            raise CustomException(e, sys)                    

    def _compile_model(self, model):
        """Compile the model with loss function and optimizer."""
        try:
            optimizer = self._get_optimizer()
            loss_function = self._get_loss_function()
            
            model.compile(optimizer=optimizer,
                        loss=loss_function,
                        metrics=["accuracy"])
            return model
        except Exception as e:
            logger.error(f"Error compiling model: {e}")
            raise CustomException(e, sys)


    def _get_optimizer(self):
        """Return the optimizer based on config."""
        optimizers = {
            "adam": tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            "sgd": tf.keras.optimizers.SGD(learning_rate=self.config.learning_rate),
            "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=self.config.learning_rate),
            "adamax": tf.keras.optimizers.Adamax(learning_rate=self.config.learning_rate),
        }
        return optimizers.get(self.config.optimizer.lower(), tf.keras.optimizers.Adam(self.config.learning_rate))

    def _get_loss_function(self):
        """Return the loss function based on config."""
        loss_functions = {
            "sparse_categorical_crossentropy": tf.keras.losses.SparseCategoricalCrossentropy(),
            "categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy(),
            "binary_crossentropy": tf.keras.losses.BinaryCrossentropy(),
        }
        return loss_functions.get(self.config.loss_function.lower(), tf.keras.losses.SparseCategoricalCrossentropy())
    
    
    def _build_custom_cnn(self):
        """Define and build a custom CNN model."""
        try:
            model = models.Sequential([
                layers.Input(shape=self.config.input_shape),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.config.num_classes, activation='softmax')
            ])
            return self._compile_model(model)
        except Exception as e:
            logger.error(f"Error building models: {e}")
            raise CustomException(e, sys)


    def save_model_summaries(self, save_path):
        """Save model summaries to a text file in a readable format."""
        try:
            with open(save_path, "w", encoding="utf-8") as file:  # Ensure UTF-8 encoding
                for model_name, model in self.models_dict.items():
                    # Capture summary as a string
                    summary_io = io.StringIO()
                    sys.stdout = summary_io
                    model.summary()
                    sys.stdout = sys.__stdout__  # Reset stdout
                    
                    # Write to file
                    file.write(f"Model: {model_name}\n")
                    file.write("=" * 80 + "\n")
                    file.write(summary_io.getvalue() + "\n")
                    file.write("=" * 80 + "\n\n")
                logger.info(f"Model summaries saved to {save_path}")
        
        except Exception as e:
            logger.error(f"Error saving model summaries: {e}")
            raise CustomException(e, sys)
            

    def build_models(self):
        """Build and return models based on configuration."""
        for model_name in self.config.model_names:
            if model_name.lower() == "custom_cnn":
                self.models_dict[model_name] = self._build_custom_cnn()
            else:
                base_model = self._load_pretrained_model(model_name)
                self.models_dict[model_name] = self._add_classification_head(base_model)

        save_path = Path(self.config.model_summaries)
        self.save_model_summaries(save_path)
        
        return self.models_dict


    def build_models(self):
        """Build and return models based on the provided configuration."""
        try:
            logger.info("Starting model building process...")

            # Loop through the list of model names specified in the config
            for model_name in self.config.model_names:
                logger.info(f"Building model: {model_name}")

                # Check if the model is a custom CNN
                if model_name.lower() == "custom_cnn":
                    self.models_dict[model_name] = self._build_custom_cnn()
                    logger.info(f"Custom CNN model built successfully.")
                else:
                    # Load pretrained model and add classification head
                    base_model = self._load_pretrained_model(model_name)
                    self.models_dict[model_name] = self._add_classification_head(base_model)
                    logger.info(f"Pretrained model '{model_name}' modified and ready.")

            # Save model summaries to the specified path
            save_path = Path(self.config.model_summaries)
            self.save_model_summaries(save_path)
            logger.info(f"Model summaries saved to {save_path}")

            logger.info("All models built and summaries generated successfully.")
            return self.models_dict
        
        except Exception as e:
            logger.error(f"Error encountered during model building: {e}")
            raise CustomException(e, sys)