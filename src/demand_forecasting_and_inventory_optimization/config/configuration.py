from src.demand_forecasting_and_inventory_optimization.constants import *
from src.demand_forecasting_and_inventory_optimization.utils.common import read_yaml, create_directories
from src.demand_forecasting_and_inventory_optimization.entity.config_entity import (DataIngestionConfig , 
                                                                                    DataValidationConfig,
                                                                                    DataPreprocessingConfig)


class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH, schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:

        config = self.config.data_validation

        schema = self.schema

        create_directories(
            [config.root_dir]
        )

        validation_config = (DataValidationConfig(
                root_dir=Path(config.root_dir),

                unzip_data_dir=Path(config.unzip_data_dir),

                STATUS_FILE=Path(config.STATUS_FILE),

                REPORT_FILE=Path(config.REPORT_FILE),

                DRIFT_REPORT_FILE=Path(config.DRIFT_REPORT_FILE),

                STATS_REPORT_FILE=Path(config.STATS_REPORT_FILE),

                all_schema=schema.COLUMNS,

                numerical_ranges=schema.NUMERICAL_RANGES,

                categorical_values=schema.CATEGORICAL_VALUES,

                thresholds=schema.THRESHOLDS,

                high_cardinality_columns=
                schema.HIGH_CARDINALITY_COLUMNS,

                leakage_columns=
                schema.LEAKAGE_COLUMNS,

                timestamp_column=
                schema.TIMESTAMP_COLUMN,

                target_column=
                schema.TARGET_COLUMN
            )
        )

        return validation_config
    

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:

        config = self.config.data_preprocessing

        schema = self.schema

        create_directories(
            [config.root_dir]
        )

        data_preprocessing_config = DataPreprocessingConfig(

            root_dir=Path(
                config.root_dir
            ),

            input_data_path=Path(
                config.input_data_path
            ),

            output_data_path=Path(
                config.output_data_path
            ),

            preprocessing_report_path=Path(
                config.preprocessing_report_path
            ),

            target_column=
            schema.TARGET_COLUMN,

            timestamp_column=
            schema.TIMESTAMP_COLUMN,

            numerical_ranges= schema.NUMERICAL_RANGES,

            outlier_method=
            config.outlier_method,

            outlier_iqr_multiplier=
            config.outlier_iqr_multiplier,

            numerical_imputation_method=
            config.numerical_imputation_method,

            categorical_imputation_method=
            config.categorical_imputation_method,

            optimize_memory=
            config.optimize_memory,

            save_format=
            config.save_format
        ) 
        return data_preprocessing_config