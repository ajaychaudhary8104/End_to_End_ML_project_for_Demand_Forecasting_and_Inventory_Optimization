from src.demand_forecasting_and_inventory_optimization.config.configuration import ConfigurationManager
from src.demand_forecasting_and_inventory_optimization.components.data_preprocessing import DataPreprocessing
from src.demand_forecasting_and_inventory_optimization import logger

STAGE_NAME = "DATA PREPROCESSING STAGE"


class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):

        config = (ConfigurationManager())

        preprocessing_config = (config.get_data_preprocessing_config())

        preprocessing = (DataPreprocessing(preprocessing_config))

        preprocessing.initiate_data_preprocessing()

if __name__ == "__main__":
    try:

        logger.info(
            f">>>>>> stage {STAGE_NAME} started <<<<<<"
        )

        obj = DataPreprocessingTrainingPipeline()

        obj.main()

        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<"
        )

    except Exception as e:

        logger.exception(e)

        raise e        