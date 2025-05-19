import os
from pathlib import Path
from src.metal_defect_detection_cnn_pipeline.utils.custom_logging import logger, project_name

# List of file paths to be checked and created if necessary
list_of_files = [
    # Version control and CI/CD
    # .gitkeep is used to ensure empty directories are tracked by Git
    ".github/workflows/.gitkeep",  # Placeholder for GitHub Actions workflow files
    ".gitignore",  # Specifies files and directories to exclude from version control

    # Source code directory and structure
    f"src/{project_name}/__init__.py",  # Marks the package as a Python module
    f"src/{project_name}/components/__init__.py",  # Module for modular components
    f"src/{project_name}/utils/__init__.py",  # Module for utility functions
    f"src/{project_name}/utils/common.py",  # Shared utility functions
    f"src/{project_name}/config/__init__.py",  # Module for configuration management
    f"src/{project_name}/config/configuration.py",  # Handles configurations and parameters
    f"src/{project_name}/pipeline/__init__.py",  # Module for data pipelines
    f"src/{project_name}/entity/__init__.py",  # Module for entity definitions
    f"src/{project_name}/entity/config_entity.py",  # Core data structure definitions
    f"src/{project_name}/constants/__init__.py",  # Centralized constants for the application
    f"src/{project_name}/constants/constant_values.py",
    f"src/{project_name}/constants/model_definitions.py",

    # Configuration and parameter management
    "config/config.yaml",  # General configuration file for the project
    "params.yaml",  # Parameters for machine learning or processing
    "schema.yaml",  # Data schema definitions or validation rules

    # Main entry point
    "main.py",  # Entry point for running the application

    # Deployment
    "Dockerfile",  # Docker configuration for containerizing the project
    "setup.py",  # Package configuration for installing or distributing the project
    "st_app_metal_defect_detection_cnn_predict.py",
    "st_app_metal_defect_detection_cnn_train_predict.py",

    # Experimentation and research
    "research/research.ipynb",  # Jupyter notebook for exploratory data analysis

    # Templates and assets
    #"templates/index.html",  # HTML template for the application (if applicable)

    # Dependency management
    "requirements.txt",  # List of dependencies for the project

    # For EDA
    "metal_surface_defect_detection_cnn.ipynb",

    # Documentation
#    "README.md",  # Project overview and instructions
#    "LICENSE",  # Legal licensing information
#    "CONTRIBUTING.md",  # Guidelines for contributors to the project

    # Testing
#    "tests/__init__.py",  # Marks the tests directory as a Python module
#    "tests/test_config.py",  # Test cases for configuration
#    "tests/test_pipeline.py",  # Test cases for pipeline functionality

    # Data directories
#    "data/raw/.gitkeep",  # Placeholder for raw data files
#    "data/processed/.gitkeep",  # Placeholder for processed data files

    # Examples and notebooks
#    "examples/example_script.py",  # Example Python script for demonstration
#    "notebooks/example_notebook.ipynb"  # Example notebook for demonstration
]

# Iterate over the list of files
for file_path in list_of_files:
    file_path = Path(file_path)  # Convert file path to Path object
    directory_name = file_path.parent  # Get the directory name from the file path
    file_name = file_path.name  # Get the file name from the file path
    
    # Create directory if directory name is not empty and if it doesn't exist
    if directory_name and not directory_name.exists():
        os.makedirs(directory_name, exist_ok=True)  # Create the directory if it doesn't exist
        logger.info(f"Created directory {directory_name} for the file: {file_name}")

    # Create empty file if it doesn't exist or is empty
    if not file_path.exists() or file_path.stat().st_size == 0:
        file_path.touch()  # Create an empty file
        logger.info(f"Created empty file: {file_path}")
    else:
        logger.info(f"File already exists: {file_path}")