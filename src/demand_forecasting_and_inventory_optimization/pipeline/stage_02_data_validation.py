from src.demand_forecasting_and_inventory_optimization.config.configuration import ConfigurationManager
from src.demand_forecasting_and_inventory_optimization.components.data_validation import DataValidation
from src.demand_forecasting_and_inventory_optimization import logger


STAGE_NAME = "DATA VALIDATION STAGE"

class DataValidationTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        validation_config = (
            config.get_data_validation_config()
        )

        validation = DataValidation(
            config=validation_config
        )

        validation_status = (
            validation.initiate_data_validation()
        )

        if not validation_status:

            raise Exception(
                "Data Validation Failed"
            )

if __name__ == "__main__":
    try:

        logger.info(
            f">>>>>> stage {STAGE_NAME} started <<<<<<"
        )

        obj = (
            DataValidationTrainingPipeline()
        )

        obj.main()

        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<"
        )

    except Exception as e:

        logger.exception(e)

        raise e