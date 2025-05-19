"""
File: constant_values.py
Purpose: Define constant file paths for configuration, schema, and parameter files.
This file defines constant file paths used throughout the project.
These constants ensure consistent and centralized file path management.
"""

from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

# Constant file paths
config_yaml_filepath = Path('config/config.yaml')
schema_yaml_filepath = Path('schema.yaml')
params_yaml_filepath = Path('params.yaml')

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")