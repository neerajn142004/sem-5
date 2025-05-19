# 🏭 **Metal Surface Defect Detection Using CNN** 🔍

## 🚀 **Project Overview**

This project aims to develop a deep learning-based system to detect and classify defects on metal surfaces using convolutional neural networks (CNN). The solution improves quality control by identifying six defect types from the NEU Metal Surface Defects dataset.

---

## 🛠️ **Workflow**

### 1️⃣ **Problem Statement**

- Detect and classify metal surface defects to improve manufacturing quality.
- Leverage CNN models to automate defect identification with high accuracy.

---

### 2️⃣ **Data Collection**

- Dataset: **NEU Metal Surface Defects Dataset**
- Contains six defect types:
  - **Crazing**  
  - **Inclusion**  
  - **Patches**  
  - **Pitted Surface**  
  - **Rolled-in Scale**  
  - **Scratches**

---

### 3️⃣ **Exploratory Data Analysis (EDA)**

- Visualized sample images from each defect category.
- Analyzed class distributions to ensure balanced representation.
- Identified the need for augmentation to address any imbalance.

---

### 4️⃣ **Data Preprocessing**

- Resized images to a fixed size.
- Normalized pixel values for faster convergence.
- Applied data augmentation to increase dataset diversity and improve model generalization.

---

### 5️⃣ **Model Building**

- Built a custom CNN model from scratch.
- Used pretrained models for transfer learning:
  - **VGG16**  
  - **ResNet50**  
  - **MobileNetV2**  
  - **EfficientNetB0**

---

### 6️⃣ **Preparing Callbacks**

- Implemented the following callbacks:
  - **Early Stopping:** Monitors validation loss and stops training if no improvement.
  - **Model Checkpoint:** Saves the best model weights during training.
  - **Learning Rate Scheduler:** Adjusts learning rate dynamically to optimize convergence.

---

### 7️⃣ **Base Model Training, Evaluation, and Selecting the Best Performing Model**

- Trained multiple CNN models and evaluated their performance.
- Used evaluation metrics such as:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
- Identified the best-performing base model based on validation metrics.

---

### 8️⃣ **Hyperparameter Tuning**

- Tuned key hyperparameters (learning rate, batch size, number of filters, etc.) to improve classification accuracy.

---

### 9️⃣ **Prediction for a Single Image**

- Developed a function to load and preprocess a single image.
- Generated predictions with confidence scores for uploaded images.

---

### 🔟 **Building the Application**

- Designed a **Streamlit** web application allowing users to:
  - Upload images for defect classification.
  - View classification results with confidence scores.

---

### 1️⃣1️⃣ **Deployment**

- Deployed the application using **Streamlit**.
- Hosted the application on **Render** for real-time accessibility and interaction.

---

## 📂 **Project Directory Structure**
```
└── metal-surface-defect-detection-cnn
    ├── .github
    │   └── workflows
    │       ├── .gitkeep
    │       └── docker-build-push.yml
    ├── artifacts
    │   ├── af_01_data_ingestion
    │   ├── af_02_data_preprocessing
    │   │   ├── resized_images_224
    │   │   │   ├── test
    │   │   │   │   ├── Crazing
    │   │   │   │   │   └── 12 *.bmp files in this folder
    │   │   │   │   ├── Inclusion
    │   │   │   │   │   └── 12 *.bmp files in this folder
    │   │   │   │   ├── Patches
    │   │   │   │   │   └── 12 *.bmp files in this folder
    │   │   │   │   ├── Pitted
    │   │   │   │   │   └── 12 *.bmp files in this folder
    │   │   │   │   ├── Rolled
    │   │   │   │   │   └── 12 *.bmp files in this folder
    │   │   │   │   └── Scratches
    │   │   │   │       └── 12 *.bmp files in this folder
    │   │   │   ├── train
    │   │   │   │   ├── Crazing
    │   │   │   │   │   └── 276 *.bmp files in this folder
    │   │   │   │   ├── Inclusion
    │   │   │   │   │   └── 276 *.bmp files in this folder
    │   │   │   │   ├── Patches
    │   │   │   │   │   └── 276 *.bmp files in this folder
    │   │   │   │   ├── Pitted
    │   │   │   │   │   └── 276 *.bmp files in this folder
    │   │   │   │   ├── Rolled
    │   │   │   │   │   └── 276 *.bmp files in this folder
    │   │   │   │   └── Scratches
    │   │   │   │       └── 276 *.bmp files in this folder
    │   │   │   └── valid
    │   │   │       ├── Crazing
    │   │   │       │   └── 12 *.bmp files in this folder
    │   │   │       ├── Inclusion
    │   │   │       │   └── 12 *.bmp files in this folder
    │   │   │       ├── Patches
    │   │   │       │   └── 12 *.bmp files in this folder
    │   │   │       ├── Pitted
    │   │   │       │   └── 12 *.bmp files in this folder
    │   │   │       ├── Rolled
    │   │   │       │   └── 12 *.bmp files in this folder
    │   │   │       └── Scratches
    │   │   │           └── 12 *.bmp files in this folder
    │   │   └── pre_processor.joblib
    │   ├── af_03_dataset_loader
    │   │   ├── test_ds
    │   │   │   ├── 12758950803948552776
    │   │   │   │   └── 00000000.shard
    │   │   │   │       └── 00000000.snapshot
    │   │   │   ├── 3258227067584415661
    │   │   │   │   └── 00000000.shard
    │   │   │   │       └── 00000000.snapshot
    │   │   │   ├── 7047874789896679519
    │   │   │   │   └── 00000000.shard
    │   │   │   │       └── 00000000.snapshot
    │   │   │   ├── dataset_spec.pb
    │   │   │   └── snapshot.metadata
    │   │   ├── train_ds
    │   │   │   ├── 10147724497377962368
    │   │   │   │   └── 00000000.shard
    │   │   │   │       └── 00000000.snapshot
    │   │   │   ├── 17286823798888458143
    │   │   │   │   └── 00000000.shard
    │   │   │   │       └── 00000000.snapshot
    │   │   │   ├── 17928277091804314537
    │   │   │   │   └── 00000000.shard
    │   │   │   │       └── 00000000.snapshot
    │   │   │   ├── dataset_spec.pb
    │   │   │   └── snapshot.metadata
    │   │   ├── valid_ds
    │   │   │   ├── 17510438350364033970
    │   │   │   │   └── 00000000.shard
    │   │   │   │       └── 00000000.snapshot
    │   │   │   ├── 54059744685680913
    │   │   │   │   └── 00000000.shard
    │   │   │   │       └── 00000000.snapshot
    │   │   │   ├── 6620044022471998455
    │   │   │   │   └── 00000000.shard
    │   │   │   │       └── 00000000.snapshot
    │   │   │   ├── dataset_spec.pb
    │   │   │   └── snapshot.metadata
    │   │   └── class_labels.json
    │   ├── af_04_model_builder
    │   │   └── model_summaries.txt
    │   ├── af_05_prepare_callbacks
    │   │   ├── checkpoint_dir
    │   │   │   ├── base_custom_cnn.h5
    │   │   │   ├── base_vgg16.h5
    │   │   │   └── tuned_vgg16.h5
    │   │   └── tensorboard_log_dir
    │   │       ├── base_custom_cnn
    │   │       │   └── tb_logs_at_2025-03-25-17-50-32
    │   │       │       ├── train
    │   │       │       │   └── events.out.tfevents.1742925032.DESKTOP-H7FFN3E.12720.2.v2
    │   │       │       └── validation
    │   │       │           └── events.out.tfevents.1742925079.DESKTOP-H7FFN3E.12720.3.v2
    │   │       ├── base_vgg16
    │   │       │   └── tb_logs_at_2025-03-25-17-47-05
    │   │       │       ├── train
    │   │       │       │   └── events.out.tfevents.1742924825.DESKTOP-H7FFN3E.12720.0.v2
    │   │       │       └── validation
    │   │       │           └── events.out.tfevents.1742924993.DESKTOP-H7FFN3E.12720.1.v2
    │   │       └── tuned_vgg16
    │   │           └── tb_logs_at_2025-03-25-17-51-27
    │   │               ├── 0
    │   │               │   └── execution0
    │   │               │       ├── train
    │   │               │       │   └── events.out.tfevents.1742925090.DESKTOP-H7FFN3E.12720.5.v2
    │   │               │       ├── validation
    │   │               │       │   └── events.out.tfevents.1742925247.DESKTOP-H7FFN3E.12720.6.v2
    │   │               │       └── events.out.tfevents.1742925088.DESKTOP-H7FFN3E.12720.4.v2
    │   │               └── 1
    │   │                   └── execution0
    │   │                       ├── train
    │   │                       │   └── events.out.tfevents.1742925257.DESKTOP-H7FFN3E.12720.8.v2
    │   │                       ├── validation
    │   │                       │   └── events.out.tfevents.1742925409.DESKTOP-H7FFN3E.12720.9.v2
    │   │                       └── events.out.tfevents.1742925256.DESKTOP-H7FFN3E.12720.7.v2
    │   ├── af_06_model_tranier_evaluation
    │   │   ├── base_models_metrics.json
    │   │   ├── base_model_custom_cnn_valid_ds_confusion_matrix.jpg
    │   │   ├── base_model_vgg16_valid_ds_confusion_matrix.jpg
    │   │   └── best_base_model.json
    │   └── af_07_hyperparameter_tuning
    │       ├── tuned_model.h5
    │       ├── tuned_models_metrics.json
    │       ├── tuned_model_test_confusion_matrix.jpg
    │       └── tuned_model_validation_confusion_matrix.jpg
    ├── config
    │   ├── config.yaml
    │   └── __init__.py
    ├── logs
    │   └── metal_defect_detection_cnn_pipeline.log
    ├── research
    │   └── research.ipynb
    ├── src
    │   ├── metal_defect_detection_cnn_pipeline
    │   │   ├── components
    │   │   │   ├── component_01_data_ingestion.py
    │   │   │   ├── component_02_data_preprocessing.py
    │   │   │   ├── component_03_dataset_loader.py
    │   │   │   ├── component_04_model_builder.py
    │   │   │   ├── component_05_prepare_callbacks.py
    │   │   │   ├── component_06_model_training_evaluation.py
    │   │   │   ├── component_07_hyper_tuning.py
    │   │   │   └── __init__.py
    │   │   ├── config
    │   │   │   ├── configuration.py
    │   │   │   └── __init__.py
    │   │   ├── constants
    │   │   │   ├── constant_values.py
    │   │   │   └── __init__.py
    │   │   ├── entity
    │   │   │   ├── config_entity.py
    │   │   │   └── __init__.py
    │   │   ├── pipeline
    │   │   │   ├── stage_01_data_ingestion.py
    │   │   │   ├── stage_02_data_preprocessing.py
    │   │   │   ├── stage_03_dataset_loader.py
    │   │   │   ├── stage_04_model_builder.py
    │   │   │   ├── stage_06_model_training_evaluation.py
    │   │   │   ├── stage_07_hyper_tuning.py
    │   │   │   ├── stage_08_prediction_pipeline.py
    │   │   │   └── __init__.py
    │   │   ├── utils
    │   │   │   ├── common.py
    │   │   │   ├── custom_exception.py
    │   │   │   ├── custom_logging.py
    │   │   │   └── __init__.py
    │   │   └── __init__.py
    │   └── __init__.py
    ├── .dockerignore
    ├── .env
    ├── .gitignore
    ├── directory_structure.txt
    ├── Dockerfile
    ├── LICENSE
    ├── main.py
    ├── metal_surface_defect_detection_cnn.ipynb
    ├── params.yaml
    ├── README.md
    ├── requirements.txt
    ├── schema.yaml
    ├── setup.py
    ├── st_app_metal_defect_detection_cnn_predict.py
    └── template.py
```


---

## 📌 **Key Takeaways**

✅ **Automated defect classification** using CNN with high accuracy.  
✅ **Transfer learning** improves performance by leveraging pretrained architectures.  
✅ **Hyperparameter tuning** optimizes model performance.  
✅ **Streamlit-based web interface** enables easy user interaction.  
✅ **Deployed on Render** for real-world application accessibility.

---
