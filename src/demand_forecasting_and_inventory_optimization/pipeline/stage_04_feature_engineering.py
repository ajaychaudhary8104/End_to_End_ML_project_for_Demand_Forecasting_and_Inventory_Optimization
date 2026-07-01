from src.demand_forecasting_and_inventory_optimization.config.configuration import ConfigurationManager
from src.demand_forecasting_and_inventory_optimization.components.feature_engineering import FeatureEngineering
from src.demand_forecasting_and_inventory_optimization import logger


STAGE_NAME = "FEATURE ENGINEERING STAGE"

class FeatureEngineeringTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        feature_engineering_config = (
            config.get_feature_engineering_config()
        )

        feature_engineering = FeatureEngineering(
            config= feature_engineering_config
        )

        
        feature_engineering.initiate_feature_engineering()
       

if __name__ == "__main__":
    try:

        logger.info(
            f">>>>>> stage {STAGE_NAME} started <<<<<<"
        )

        obj = (
            FeatureEngineeringTrainingPipeline()
        )

        obj.main()

        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<"
        )

    except Exception as e:

        logger.exception(e)

        raise e