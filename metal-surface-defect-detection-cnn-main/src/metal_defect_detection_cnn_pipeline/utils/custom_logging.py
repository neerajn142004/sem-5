import logging
import sys
from pathlib import Path
import io

# Define the project name
project_name = "metal_defect_detection_cnn_pipeline"


# Define the directory for storing logs
log_dir = Path("logs")

# Check if the logs directory exists before creating it
if not log_dir.exists():
    log_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

# Define the log file path
log_file = log_dir / f"{project_name}.log"


if hasattr(sys.stdout, "buffer"):  # Check if sys.stdout has 'buffer' attribute
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set up the logging configuration
logging.basicConfig(
    level = logging.INFO,  # Set the logging level to INFO
    format = '%(asctime)s - %(levelname)s - %(module)s - %(message)s',  # Format of the log message
    datefmt = '%Y-%m-%d %H:%M:%S',  # Date format in the log
    handlers=[  # Define the handlers for logging: to both a file and the console
        logging.FileHandler(log_file),  # Log to a file
        logging.StreamHandler(sys.stdout)  # Log to the console
    ]
)

logger = logging.getLogger(project_name)