import streamlit as st
import os
import json
from PIL import Image

from src.metal_defect_detection_cnn_pipeline.pipeline.stage_08_prediction_pipeline import PredictionPipeline
from src.metal_defect_detection_cnn_pipeline.utils.custom_logging import logger

# Paths
TEST_FOLDER_PATH = "artifacts/af_01_data_ingestion/input_data/neu_metal_surface_defects_data/test/"
LABELS_PATH = "artifacts/af_03_dataset_loader/class_labels.json"

# Load class labels
with open(LABELS_PATH, "r") as f:
    class_labels = json.load(f)

# Function to make predictions
def make_prediction(image_path):
    try:
        logger.info("Starting prediction process.")
        # Initialize the prediction pipeline
        pipeline = PredictionPipeline()
        # Make prediction using the input image
        prediction = pipeline.predict(image_path)
        logger.info(f"Prediction successful: {prediction}")
        return prediction
    except Exception as e:
        logger.exception("Exception occurred during prediction.")
        st.error(f"An error occurred during prediction: {e}")
        return None


# Streamlit App Layout
def main():
    st.set_page_config(page_title="Metal Defect Detection", layout="wide")

    # Header
    st.title("🔎 Metal Defect Detection Using CNN")
    st.write("""
        ### Upload an image or drag images from the test set folders to detect defects.
    """)

    # Sidebar - Display test set folders with images
    st.sidebar.title("📁 Test Set Preview")
    st.sidebar.markdown("### Drag and drop files or choose images below:")

    # Display class folders with sample images
    for label in class_labels:
        class_path = os.path.join(TEST_FOLDER_PATH, label)
        if os.path.exists(class_path):
            st.sidebar.markdown(f"**{label}**")
            images = [f for f in os.listdir(class_path) if f.endswith(".bmp")][:3]

            # Show sample images with hover and enlarge option
            cols = st.sidebar.columns(len(images))
            for i, img_name in enumerate(images):
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path)

                with cols[i]:
                    st.image(img, use_container_width=True)
                    if st.button(f"{img_name}", key=f"preview_{img_name}"):
                        st.image(img, caption=f"Preview: {img_name}", use_container_width=True)

    # File uploader and drag-and-drop section
    st.markdown("### 📸 Drag images here or upload manually:")
    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)

    dragged_files = []
    if uploaded_files:
        dragged_files.extend(uploaded_files)

    # Process dragged files from sidebar
    for label in class_labels:
        class_path = os.path.join(TEST_FOLDER_PATH, label)
        if os.path.exists(class_path):
            for img_name in os.listdir(class_path):
                if f"drag_{img_name}" in st.session_state:
                    dragged_files.append(os.path.join(class_path, img_name))

    # Process and display uploaded or dragged images
    if dragged_files:
        for file in dragged_files:
            if isinstance(file, str):
                # Image dragged from sidebar
                img_path = file
                img = Image.open(img_path)
            else:
                # Uploaded image
                img_path = f"temp_{file.name}"
                img = Image.open(file)
                img.save(img_path)

            # Display image preview
            st.image(img, caption=f"Uploaded: {file.name if hasattr(file, 'name') else os.path.basename(file)}",
                     use_container_width=True)

            # Button to make predictions for each image
            if st.button(f"Predict for {file.name if hasattr(file, 'name') else os.path.basename(file)}"):
                prediction = make_prediction(img_path)

                if prediction is not None:
                    st.success(f"✅ Prediction: {prediction}")

# Run the app
if __name__ == "__main__":
    main()