# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy all the files from the current directory into the container's /app/ directory
COPY . /app/
#COPY artifacts /app/artifacts

# List the files copied to the container with modification date, time, and size
RUN echo "Files copied to /app/:" && ls -lR /app/

# Copy only the necessary files
#COPY app.py /app/
#COPY requirements.txt /app/
#COPY artifacts/data_preprocessing/preprocessor.joblib /app/artifacts/data_preprocessing/
#COPY artifacts/train_evaluate_model/model.joblib /app/artifacts/train_evaluate_model/

# Install dependencies

RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "st_app_metal_defect_detection_cnn_predict.py"]