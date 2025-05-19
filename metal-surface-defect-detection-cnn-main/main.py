import sys
from src.metal_defect_detection_cnn_pipeline.utils.custom_logging import logger
from src.metal_defect_detection_cnn_pipeline.utils.custom_exception import CustomException

from src.metal_defect_detection_cnn_pipeline.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.metal_defect_detection_cnn_pipeline.pipeline.stage_02_data_preprocessing import DataPreprocessingPipeline
from src.metal_defect_detection_cnn_pipeline.pipeline.stage_03_dataset_loader import DatasetLoaderPipeline
from src.metal_defect_detection_cnn_pipeline.pipeline.stage_04_model_builder import ModelBuilderPipeline
from src.metal_defect_detection_cnn_pipeline.pipeline.stage_06_model_training_evaluation import ModelTrainerEvaluationPipeline
from src.metal_defect_detection_cnn_pipeline.pipeline.stage_07_hyper_tuning import ModelHyperparameterTuningPipeline

STAGE_NAME = 'Data Ingestion Stage'
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx=============x")
except Exception as e:
    logger.exception(e)
    raise CustomException(e,sys)

STAGE_NAME = 'Data Preprocessing Stage'
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = DataPreprocessingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx=============x")
except Exception as e:
    logger.exception(e)
    raise CustomException(e,sys)

STAGE_NAME = 'Dataset Loader Stage'
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = DatasetLoaderPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx=============x")
except Exception as e:
    logger.exception(e)
    raise CustomException(e,sys)


STAGE_NAME = 'Model Builder Stage'
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = ModelBuilderPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx=============x")
except Exception as e:
    logger.exception(e)
    raise CustomException(e,sys)


STAGE_NAME = 'Model Trainer Evaluation Stage'
try:
    logger.info(f">>>>>>> Stage: {STAGE_NAME} started <<<<<<<")
    obj = ModelTrainerEvaluationPipeline()
    obj.main()
    logger.info(f">>>>>>> Stage: {STAGE_NAME} completed <<<<<<<\n\nx=============x")
except Exception as e:
    logger.exception(f"An error occurred in stage: {STAGE_NAME}.")
    raise CustomException(e, sys)


STAGE_NAME = 'Model Hyperparameter Tuning Stage'
try:
    logger.info(f">>>>>>> Stage: {STAGE_NAME} started <<<<<<<")
    obj = ModelHyperparameterTuningPipeline()
    obj.main()
    logger.info(f">>>>>>> Stage: {STAGE_NAME} completed <<<<<<<\n\nx=============x")
except Exception as e:
    logger.exception(f"An error occurred in stage: {STAGE_NAME}.")
    raise CustomException(e, sys)