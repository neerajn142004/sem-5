"""
File: common.py

Description:
This module provides utility functions for common operations such as:
- Reading and parsing YAML files into a ConfigBox object.
- Creating directories in a safe and idempotent way.
- Saving and loading data in JSON and binary formats.
- Calculating and formatting file sizes for readability.
"""

import os
import yaml
import sys
import json
import joblib

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from box import ConfigBox
from box.exceptions import BoxValueError

from ensure import ensure_annotations

from typing import Any

from .custom_logging import logger
from .custom_exception import CustomException

from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    make_scorer
)

# -------------------------- Reading YAML file --------------------------
# This function reads a YAML file from the specified path and returns its content 
# as a ConfigBox object. It uses type annotations to ensure type safety 
# and handles exceptions for better robustness.

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads and parses a YAML file into a ConfigBox object.
    Args:
        path_to_yaml (Path): Path to the YAML file.
    Returns:
        ConfigBox: A dictionary-like object containing the parsed YAML content.
    Raises:
        ValueError: If the YAML file is empty.
        CustomException: If any other exception occurs during file reading or parsing.
    """
    try:
        # Open the YAML file and read its contents
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)  # Safely parse the YAML file
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")  # Log success
            return ConfigBox(content)  # Return the parsed content as a ConfigBox object

    # Handle case where the YAML file is empty
    except BoxValueError:
        raise ValueError("yaml file is empty")

    # Handle any other unexpected exceptions
    except Exception as e:
        raise CustomException(e, sys)

# ----------------------- Create Directories -------------------------
# This function creates multiple directories specified in a list of paths.
# It ensures the directories exist and logs the action if verbose is enabled.

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Creates directories from a list of paths.
    Args:
        path_to_directories (list): List of directory paths to create.
        verbose (bool): If True, logs the creation of each directory.
    Returns:
        None
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)  # Create directory, avoid errors if it exists
        if verbose:
            logger.info(f"Created directory at: {path}")  # Log creation


# ----------------------- Save JSON -------------------------
# This function saves a dictionary as a JSON file at the specified path.
# The file is formatted for readability, and the action is logged.

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Saves a dictionary as a JSON file.
    Args:
        path (Path): Path to save the JSON file.
        data (dict): Data to be saved in the JSON file.
    Returns:
        None
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)  # Save JSON with indentation
    logger.info(f"JSON file saved at: {path}")  # Log save action


# ----------------------- Load JSON -------------------------
# This function reads a JSON file and returns its content as a ConfigBox object.
# The action is logged for traceability.

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Loads a JSON file and converts it to a ConfigBox.
    Args:
        path (Path): Path to the JSON file.
    Returns:
        ConfigBox: The content of the JSON file as a dictionary-like object.
    """
    with open(path) as f:
        content = json.load(f)  # Load JSON content
        content = ConfigBox(content)  # Convert to ConfigBox for easier access
    logger.info(f"JSON file loaded from: {path}")  # Log load action
    return content


# ----------------------- Save Binary -------------------------
# This function saves data as a binary file using joblib.
# The action is logged for record-keeping.

#@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Saves data to a binary file using joblib.
    Args:
        data (Any): The data to be saved.
        path (Path): Path to save the binary file.
    Returns:
        None
    """
    joblib.dump(value=data, filename=path)  # Save binary file
    logger.info(f"Binary file saved at: {path}")  # Log save action


# ----------------------- Load Binary -------------------------
# This function reads data from a binary file using joblib.
# The loaded data is returned, and the action is logged.

#@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Loads data from a binary file using joblib.
    Args:
        path (Path): Path to the binary file.
    Returns:
        Any: The loaded data.
    """
    data = joblib.load(path)  # Load binary data
    logger.info(f"Binary file loaded from: {path}")  # Log load action
    return data


# ----------------------- Get File Size -------------------------
# This function calculates and returns the size of a file in kilobytes (KB).
# It provides a human-readable representation of the file size.

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Returns the size of a file in kilobytes (KB).
    Args:
        path (Path): Path to the file.
    Returns:
        str: File size in kilobytes, rounded and prefixed with '~'.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)  # Get file size in KB
    return f"~ {size_in_kb} KB"  # Return formatted size