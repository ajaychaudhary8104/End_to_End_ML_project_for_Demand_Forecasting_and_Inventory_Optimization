import numpy as np
import pandas as pd
from demand_forecasting_and_inventory_optimization.entity.config_entity import FeatureEngineeringConfig
from src.demand_forecasting_and_inventory_optimization import logger
from src.demand_forecasting_and_inventory_optimization.utils.common import save_json
from src.demand_forecasting_and_inventory_optimization.entity.config_entity import FeatureEngineeringConfig
import os
from pathlib import Path



class FeatureEngineering:

    """
    Production Grade Feature Engineering
    For Demand Forecasting & Inventory Optimization

    Features:
        1. Time Features
        2. Cyclical Date Features
        3. Demand Lag Features
        4. Rolling Statistics
        5. Price Features
        6. Inventory Features
        7. Supplier Features
        8. Weather Features
        9. Economic Features
        10. Demand Trend Features
        11. Seasonal Features
        12. Inventory Risk Features
    """

    def __init__(self, config: FeatureEngineeringConfig):

        self.config = config

        self.data = pd.read_parquet(
            self.config.input_data_path
        )

        self.feature_report = {}

        logger.info(
            f"Dataset Loaded "
            f"Shape={self.data.shape}"
        )

    # =====================================================
    # TIME FEATURES
    # =====================================================

    def create_time_features(self):

        self.data["date"] = pd.to_datetime(
            self.data["date"]
        )

        self.data["quarter"] = (
            self.data["date"]
            .dt.quarter
        )

        self.data["dayofyear"] = (
            self.data["date"]
            .dt.dayofyear
        )

        self.data["is_month_start"] = (
            self.data["date"]
            .dt.is_month_start
            .astype(int)
        )

        self.data["is_month_end"] = (
            self.data["date"]
            .dt.is_month_end
            .astype(int)
        )

        logger.info(
            "Time features created"
        )

    # =====================================================
    # CYCLICAL FEATURES
    # =====================================================

    def create_cyclical_features(self):

        self.data["month_sin"] = np.sin(
            2 *
            np.pi *
            self.data["month"] /
            12
        )

        self.data["month_cos"] = np.cos(
            2 *
            np.pi *
            self.data["month"] /
            12
        )

        self.data["dayofweek_sin"] = np.sin(
            2 *
            np.pi *
            self.data["dayofweek"] /
            7
        )

        self.data["dayofweek_cos"] = np.cos(
            2 *
            np.pi *
            self.data["dayofweek"] /
            7
        )

        logger.info(
            "Cyclical features created"
        )

    # =====================================================
    # PRICE FEATURES
    # =====================================================

    def create_price_features(self):

        self.data["profit_per_unit"] = (

            self.data["unit_price"]

            -

            self.data["unit_cost"]
        )

        self.data["markup_ratio"] = (

            self.data["unit_price"]

            /

            (
                self.data["unit_cost"]
                + 1e-6
            )
        )

        self.data["discount_impact"] = (

            self.data["discount_pct"]

            *

            self.data["unit_price"]
        )

        logger.info(
            "Price features created"
        )

    # =====================================================
    # INVENTORY FEATURES
    # =====================================================

    def create_inventory_features(self):

        self.data["inventory_position"] = (

            self.data["stock_on_hand"]

            +

            self.data["stock_in_transit"]
        )

        self.data["inventory_gap"] = (

            self.data["inventory_position"]

            -

            self.data["units_sold"]
        )

        self.data["backorder_ratio"] = (

            self.data["backorder_qty"]

            /

            (
                self.data["stock_on_hand"]
                + 1
            )
        )

        self.data["inventory_turnover_proxy"] = (

            self.data["units_sold"]

            /

            (
                self.data["stock_on_hand"]
                + 1
            )
        )

        logger.info(
            "Inventory features created"
        )

    # =====================================================
    # SUPPLIER FEATURES
    # =====================================================

    def create_supplier_features(self):

        self.data["supplier_efficiency"] = (

            self.data["supplier_rating"]

            /

            (
                self.data["avg_lead_time"]
                + 1
            )
        )

        self.data["lead_time_variability"] = (

            self.data["lead_time_std"]

            /

            (
                self.data["avg_lead_time"]
                + 1
            )
        )

        logger.info(
            "Supplier features created"
        )

    # =====================================================
    # WEATHER FEATURES
    # =====================================================

    def create_weather_features(self):

        self.data["heat_index_proxy"] = (

            self.data["temperature"]

            *

            (
                self.data["humidity"]
                /
                100
            )
        )

        self.data["rainfall_intensity"] = (

            self.data["rainfall"]

            /

            (
                self.data["humidity"]
                + 1
            )
        )

        logger.info(
            "Weather features created"
        )

    # =====================================================
    # ECONOMIC FEATURES
    # =====================================================

    def create_economic_features(self):

        self.data["inflation_fuel_index"] = (

            self.data["inflation_rate"]

            *

            self.data["fuel_price"]
        )

        self.data["consumer_spending_proxy"] = (

            self.data["consumer_index"]

            /

            (
                self.data["inflation_rate"]
                + 1
            )
        )

        logger.info(
            "Economic features created"
        )

    # =====================================================
    # DEMAND LAG FEATURES
    # =====================================================

    def create_lag_features(self):

        group_cols = [

            "product_id",

            "store_id"
        ]

        self.data.sort_values(
            [
                "product_id",
                "store_id",
                "date"
            ],
            inplace=True
        )

        for lag in [1, 7, 14, 28]:

            self.data[
                f"lag_{lag}"
            ] = (

                self.data
                .groupby(group_cols)
                ["units_sold"]
                .shift(lag)
            )

        logger.info(
            "Lag features created"
        )

    # =====================================================
    # ROLLING FEATURES
    # =====================================================

    def create_rolling_features(self):

        group_cols = [

            "product_id",

            "store_id"
        ]

        self.data["rolling_mean_7"] = (

            self.data
            .groupby(group_cols)
            ["units_sold"]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(7)
                .mean()
            )
        )

        self.data["rolling_mean_14"] = (

            self.data
            .groupby(group_cols)
            ["units_sold"]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(14)
                .mean()
            )
        )

        self.data["rolling_std_7"] = (

            self.data
            .groupby(group_cols)
            ["units_sold"]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(7)
                .std()
            )
        )

        logger.info(
            "Rolling features created"
        )

    # =====================================================
    # DEMAND TREND FEATURES
    # =====================================================

    def create_demand_trend_features(self):

        self.data["demand_growth_7"] = (

            (
                self.data["lag_1"]

                -

                self.data["lag_7"]
            )

            /

            (
                self.data["lag_7"]
                + 1
            )
        )

        self.data["demand_growth_28"] = (

            (
                self.data["lag_1"]

                -

                self.data["lag_28"]
            )

            /

            (
                self.data["lag_28"]
                + 1
            )
        )

        logger.info(
            "Demand trend features created"
        )

    # =====================================================
    # REVENUE FEATURES
    # =====================================================

    def create_revenue_features(self):

        self.data["revenue_per_unit"] = (

            self.data["revenue"]

            /

            (
                self.data["units_sold"]
                + 1
            )
        )

        self.data["expected_profit"] = (

            self.data["units_sold"]

            *

            (
                self.data["unit_price"]
                -
                self.data["unit_cost"]
            )
        )

        logger.info(
            "Revenue features created"
        )

    # =====================================================
    # STOCKOUT FEATURES
    # =====================================================

    def create_stockout_features(self):

        self.data["stockout_risk"] = (

            self.data["units_sold"]

            >

            self.data["stock_on_hand"]
        ).astype(int)

        self.data["inventory_health_score"] = (

            self.data["stock_on_hand"]

            /

            (
                self.data["units_sold"]
                + 1
            )
        )

        logger.info(
            "Stockout features created"
        )

    # =====================================================
    # CATEGORY AGGREGATE FEATURES
    # =====================================================

    def create_category_aggregates(self):

        category_mean = (

            self.data
            .groupby("category")
            ["units_sold"]
            .transform("mean")
        )

        self.data[
            "category_avg_demand"
        ] = category_mean

        store_mean = (

            self.data
            .groupby("store_id")
            ["units_sold"]
            .transform("mean")
        )

        self.data[
            "store_avg_demand"
        ] = store_mean

        logger.info(
            "Aggregate features created"
        )

    # =====================================================
    # HANDLE GENERATED NANS
    # =====================================================

    def handle_generated_nans(self):

        numeric_cols = (

            self.data
            .select_dtypes(
                include=np.number
            )
            .columns
        )

        self.data[
            numeric_cols
        ] = self.data[
            numeric_cols
        ].fillna(0)

        logger.info(
            "Generated NaNs handled"
        )

    # =====================================================
    # SAVE FEATURES
    # =====================================================

    def save_features(self):

        os.makedirs(
            os.path.dirname(
                self.config.output_data_path
            ),
            exist_ok=True
        )

        self.data.to_parquet(
            self.config.output_data_path,
            index=False
        )

        self.feature_report["rows"] = int(len(self.data))

        self.feature_report["columns"] = int(len(self.data.columns))

        

        self.feature_report[
            "generated_features"
        ] = int(
            len(
                self.data.columns
            )
            -
            self.config.original_feature_count
        )

        save_json(
            Path(
                self.config.feature_report_path
            ),
            self.feature_report
        )

        logger.info(
            "Feature engineering completed"
        )

    # =====================================================
    # MAIN PIPELINE
    # =====================================================

    def initiate_feature_engineering(self):

        logger.info(
            "Starting Feature Engineering"
        )

        self.create_time_features()

        self.create_cyclical_features()

        self.create_price_features()

        self.create_inventory_features()

        self.create_supplier_features()

        self.create_weather_features()

        self.create_economic_features()

        self.create_lag_features()

        self.create_rolling_features()

        self.create_demand_trend_features()

        self.create_revenue_features()

        self.create_stockout_features()

        self.create_category_aggregates()

        self.handle_generated_nans()

        self.save_features()

        logger.info(
            "Feature Engineering Pipeline Completed"
        )

        return (
            self.config.output_data_path
        )