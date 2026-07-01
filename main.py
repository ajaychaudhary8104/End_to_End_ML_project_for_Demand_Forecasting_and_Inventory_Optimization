from src.demand_forecasting_and_inventory_optimization import logger
from src.demand_forecasting_and_inventory_optimization.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.demand_forecasting_and_inventory_optimization.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.demand_forecasting_and_inventory_optimization.pipeline.stage_03_data_preprocessing import DataPreprocessingTrainingPipeline

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Validation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_validation = DataValidationTrainingPipeline()
   data_validation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Preprocessing stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   data_preprocessing = DataPreprocessingTrainingPipeline()
   data_preprocessing.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
