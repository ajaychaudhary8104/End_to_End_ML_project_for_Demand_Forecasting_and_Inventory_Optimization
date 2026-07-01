import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from src.demand_forecasting_and_inventory_optimization import logger
from src.demand_forecasting_and_inventory_optimization.utils.common import save_json
from src.demand_forecasting_and_inventory_optimization.entity.config_entity import DataValidationConfig


class DataValidation:

    def __init__(self,config: DataValidationConfig):

        self.config = config

        self.data = self.create_merged_dataframe_and_save()

        self.validation_report = {}

        self.dataset_statistics = {}

        self.drift_report = {}

        logger.info(
            f"Dataset Loaded Successfully "
            f"Shape={self.data.shape}"
        )

    # =====================================================
    # CREATE MASTER DATAFRAME
    # =====================================================
    def create_merged_dataframe_and_save(self) -> pd.DataFrame:
        data_dir = self.config.unzip_data_dir
        sales_df = pd.read_parquet(os.path.join(data_dir, 'sales.parquet'))
        inventory_df = pd.read_parquet(os.path.join(data_dir, 'inventory.parquet'))
        products_df = pd.read_parquet(os.path.join(data_dir, 'products.parquet'))
        stores_df = pd.read_parquet(os.path.join(data_dir, 'stores.parquet'))
        weather_df = pd.read_parquet(os.path.join(data_dir, 'weather.parquet'))
        economics_df = pd.read_parquet(os.path.join(data_dir, 'economics.parquet'))
        promotions_df = pd.read_parquet(os.path.join(data_dir, 'promotions.parquet'))
        suppliers_df = pd.read_parquet(os.path.join(data_dir, 'suppliers.parquet'))
        promotions_clean = (promotions_df.groupby(["date", "product_id", "store_id"], as_index=False).agg({"discount_pct": "max"}))
        master_df = sales_df.copy()

        # Inventory
        master_df = master_df.merge(
            inventory_df,
            on=["date", "product_id", "store_id"],
            how="left",
            validate="one_to_one"
        )

        # Promotions
        master_df = master_df.merge(
            promotions_clean,
            on=["date", "product_id", "store_id"],
            how="left",
            validate="one_to_one"
        )

        # Supplier Information
        master_df = master_df.merge(
            suppliers_df,
            on="supplier_id",
            how="left",
            validate="many_to_one"
        )

        # Economic Indicators
        master_df = master_df.merge(
            economics_df,
            on="date",
            how="left",
            validate="many_to_one"
        )

        # Fill missing promotions
        master_df["discount_pct"] = master_df["discount_pct"].fillna(0)
        
        logger.info("saving merged dataframe to parquet file")
        master_df.to_parquet(
            os.path.join(self.config.unzip_data_dir, "master.parquet"),
            index=False
        )
        return master_df
    
    # =====================================================
    # REPORT HELPERS
    # =====================================================

    def _update_report(
        self,
        validation_name: str,
        status: bool,
        details=None
    ):

        self.validation_report[
            validation_name
        ] = {

            "status": status,
            "details": details
        }

    def _save_reports(self):

        save_json(
            self.config.REPORT_FILE,
            self.validation_report
        )

        save_json(
            self.config.STATS_REPORT_FILE,
            self.dataset_statistics
        )

        if len(self.drift_report) > 0:

            save_json(
                self.config.DRIFT_REPORT_FILE,
                self.drift_report
            )

    # =====================================================
    # DATASET STATISTICS
    # =====================================================

    def generate_dataset_statistics(self):

        stats = {

            "rows":
                int(
                    self.data.shape[0]
                ),

            "columns":
                int(
                    self.data.shape[1]
                ),

            "duplicates":
                int(
                    self.data
                    .duplicated()
                    .sum()
                ),

            "memory_usage_mb":
                round(
                    self.data
                    .memory_usage(
                        deep=True
                    )
                    .sum()
                    /
                    1024
                    /
                    1024,
                    2
                ),

            "date_min":
                str(
                    self.data["date"]
                    .min()
                ),

            "date_max":
                str(
                    self.data["date"]
                    .max()
                ),

            "target_stats":
                {
                    "min":
                        float(
                            self.data[
                                self.config.target_column
                            ].min()
                        ),

                    "max":
                        float(
                            self.data[
                                self.config.target_column
                            ].max()
                        ),

                    "mean":
                        float(
                            self.data[
                                self.config.target_column
                            ].mean()
                        ),

                    "std":
                        float(
                            self.data[
                                self.config.target_column
                            ].std()
                        )
                }
        }

        self.dataset_statistics = stats

        logger.info(
            "Dataset Statistics Generated"
        )

    # =====================================================
    # SCHEMA VALIDATION
    # =====================================================

    def validate_schema(self):

        expected_columns = list(
            self.config.all_schema.keys()
        )

        actual_columns = list(
            self.data.columns
        )

        missing_columns = [
            col
            for col in expected_columns
            if col not in actual_columns
        ]

        extra_columns = [
            col
            for col in actual_columns
            if col not in expected_columns
        ]

        validation_passed = (
            len(missing_columns) == 0
        )

        self._update_report(
            "Schema Validation",
            validation_passed,
            {
                "missing_columns":
                    missing_columns,

                "extra_columns":
                    extra_columns
            }
        )

        return validation_passed

    # =====================================================
    # DATATYPE VALIDATION
    # =====================================================

    def validate_datatypes(self):

        mismatches = []

        for (
            col,
            expected_dtype
        ) in self.config.all_schema.items():

            if col not in self.data.columns:
                continue

            actual_dtype = str(
                self.data[col].dtype
            )

            if (
                actual_dtype
                != expected_dtype
            ):

                mismatches.append(
                    {
                        "column":
                            col,

                        "expected":
                            expected_dtype,

                        "actual":
                            actual_dtype
                    }
                )

        validation_passed = (
            len(mismatches) == 0
        )

        self._update_report(
            "Datatype Validation",
            validation_passed,
            mismatches
        )

        return validation_passed

    # =====================================================
    # MISSING VALUES
    # =====================================================

    def validate_missing_values(self):

        threshold = (
            self.config.thresholds
            .missing_value_threshold
        )

        issues = []

        for col in self.data.columns:

            missing_ratio = (
                self.data[col]
                .isnull()
                .mean()
            )

            if (
                missing_ratio
                >
                threshold
            ):

                issues.append(
                    {
                        "column":
                            col,

                        "missing_pct":
                            round(
                                missing_ratio
                                * 100,
                                2
                            )
                    }
                )

        validation_passed = (
            len(issues) == 0
        )

        self._update_report(
            "Missing Value Validation",
            validation_passed,
            issues
        )

        return validation_passed

    # =====================================================
    # DUPLICATES
    # =====================================================

    def validate_duplicates(self):

        duplicate_count = int(
            self.data
            .duplicated()
            .sum()
        )

        duplicate_pct = (
            duplicate_count
            /
            len(self.data)
        )

        validation_passed = (
            duplicate_pct
            <=
            self.config.thresholds
            .max_duplicate_percentage
        )

        self._update_report(
            "Duplicate Validation",
            validation_passed,
            {
                "duplicate_count":
                    duplicate_count,

                "duplicate_pct":
                    round(
                        duplicate_pct * 100,
                        4
                    )
            }
        )

        return validation_passed

    # =====================================================
    # NUMERICAL RANGE
    # =====================================================

    def validate_numerical_ranges(self):

        issues = []

        for (
            col,
            limits
        ) in self.config.numerical_ranges.items():

            if col not in self.data.columns:
                continue

            invalid = self.data[
                (
                    self.data[col]
                    < limits.min
                )
                |
                (
                    self.data[col]
                    > limits.max
                )
            ]

            if len(invalid) > 0:

                issues.append(
                    {
                        "column":
                            col,

                        "violations":
                            int(
                                len(
                                    invalid
                                )
                            )
                    }
                )

        validation_passed = (
            len(issues) == 0
        )

        self._update_report(
            "Numerical Range Validation",
            validation_passed,
            issues
        )

        return validation_passed

    # =====================================================
    # CATEGORICAL VALIDATION
    # =====================================================

    def validate_categorical_values(self):

        issues = []

        for (
            col,
            allowed_values
        ) in (
            self.config
            .categorical_values
            .items()
        ):

            if col not in self.data.columns:
                continue

            invalid = self.data[
                ~self.data[col]
                .isin(
                    list(
                        allowed_values
                    )
                )
            ]

            if len(invalid) > 0:

                issues.append(
                    {
                        "column":
                            col,

                        "invalid_count":
                            len(
                                invalid
                            )
                    }
                )

        validation_passed = (
            len(issues) == 0
        )

        self._update_report(
            "Categorical Validation",
            validation_passed,
            issues
        )

        return validation_passed

    # =====================================================
    # TARGET VALIDATION
    # =====================================================

    def validate_target(self):

        target_col = (
            self.config.target_column
        )

        invalid_count = int(
            (
                self.data[target_col]
                < 0
            ).sum()
        )

        validation_passed = (
            invalid_count == 0
        )

        self._update_report(
            "Target Validation",
            validation_passed,
            {
                "negative_values":
                    invalid_count
            }
        )

        return validation_passed

    # =====================================================
    # TIMESTAMP VALIDATION
    # =====================================================

    def validate_timestamp(self):

        timestamp_col = (
            self.config.timestamp_column
        )

        timestamps = pd.to_datetime(
            self.data[timestamp_col],
            errors="coerce"
        )

        invalid_dates = int(
            timestamps.isnull().sum()
        )

        future_dates = int(
            (
                timestamps
                >
                pd.Timestamp.now()
            ).sum()
        )

        validation_passed = (
            invalid_dates == 0
            and
            future_dates == 0
        )

        self._update_report(
            "Timestamp Validation",
            validation_passed,
            {
                "invalid_dates":
                    invalid_dates,

                "future_dates":
                    future_dates
            }
        )

        return validation_passed

    # =====================================================
    # TIME SERIES CONTINUITY
    # =====================================================

    def validate_time_series_continuity(self):

        dates = pd.to_datetime(
            self.data["date"]
        )

        expected_dates = (
            pd.date_range(
                start=dates.min(),
                end=dates.max(),
                freq="D"
            )
        )

        actual_dates = (
            dates
            .drop_duplicates()
            .sort_values()
        )

        missing_dates = (
            expected_dates
            .difference(
                actual_dates
            )
        )

        validation_passed = (
            len(missing_dates) == 0
        )

        self._update_report(
            "Time Series Continuity",
            validation_passed,
            {
                "missing_dates":
                    len(
                        missing_dates
                    )
            }
        )

        return validation_passed

    # =====================================================
    # BUSINESS RULES
    # =====================================================

    def validate_business_rules(self):

        issues = []

        revenue_check = (
            self.data["units_sold"]
            *
            self.data["unit_price"]
        )

        revenue_mismatch = (
            np.abs(
                revenue_check
                -
                self.data["revenue"]
            )
            >
            0.01
        )

        if revenue_mismatch.sum() > 0:

            issues.append(
                {
                    "rule":
                        "Revenue Consistency",

                    "violations":
                        int(
                            revenue_mismatch
                            .sum()
                        )
                }
            )

        inventory_cols = [

            "stock_on_hand",

            "stock_in_transit",

            "backorder_qty"
        ]

        for col in inventory_cols:

            violations = int(
                (
                    self.data[col]
                    < 0
                ).sum()
            )

            if violations > 0:

                issues.append(
                    {
                        "rule":
                            f"{col} >= 0",

                        "violations":
                            violations
                    }
                )

        validation_passed = (
            len(issues) == 0
        )

        self._update_report(
            "Business Rule Validation",
            validation_passed,
            issues
        )

        return validation_passed

    # =====================================================
    # INVENTORY VALIDATION
    # =====================================================

    def validate_inventory_logic(self):

        stockout_rows = self.data[

            self.data[
                "units_sold"
            ]
            >
            (
                self.data[
                    "stock_on_hand"
                ]
                +
                self.data[
                    "backorder_qty"
                ]
            )
        ]

        self._update_report(
            "Inventory Validation",
            True,
            {
                "potential_stockouts":
                    len(
                        stockout_rows
                    )
            }
        )

        return True

    # =====================================================
    # PRODUCT-STORE COVERAGE
    # =====================================================

    def validate_product_store_coverage(self):

        expected = (
            self.data[
                "product_id"
            ]
            .nunique()
            *
            self.data[
                "store_id"
            ]
            .nunique()
        )

        actual = (
            self.data[
                [
                    "product_id",
                    "store_id"
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

        validation_passed = (
            actual == expected
        )

        self._update_report(
            "Product Store Coverage",
            validation_passed,
            {
                "expected":
                    expected,

                "actual":
                    actual
            }
        )

        return validation_passed

    # =====================================================
    # DATA FRESHNESS
    # =====================================================

    def validate_data_freshness(self):

        latest_date = (
            pd.to_datetime(
                self.data["date"]
            )
            .max()
        )

        self._update_report(
            "Data Freshness",
            True,
            {
                "latest_date":
                    str(
                        latest_date
                    )
            }
        )

        return True

    # =====================================================
    # INFINITE VALUES
    # =====================================================

    def validate_infinite_values(self):

        issues = []

        numeric_cols = (
            self.data
            .select_dtypes(
                include=np.number
            )
            .columns
        )

        for col in numeric_cols:

            count = int(
                np.isinf(
                    self.data[col]
                ).sum()
            )

            if count > 0:

                issues.append(
                    {
                        "column":
                            col,

                        "count":
                            count
                    }
                )

        validation_passed = (
            len(issues) == 0
        )

        self._update_report(
            "Infinite Value Validation",
            validation_passed,
            issues
        )

        return validation_passed

    # =====================================================
    # OUTLIER ANALYSIS
    # =====================================================

    def validate_outliers(self):

        outlier_report = {}

        numeric_cols = (
            self.data
            .select_dtypes(
                include=np.number
            )
            .columns
        )

        for col in numeric_cols:

            q1 = (
                self.data[col]
                .quantile(0.25)
            )

            q3 = (
                self.data[col]
                .quantile(0.75)
            )

            iqr = q3 - q1

            lower = (
                q1
                -
                1.5 * iqr
            )

            upper = (
                q3
                +
                1.5 * iqr
            )

            outlier_report[col] = int(
                (
                    (
                        self.data[col]
                        < lower
                    )
                    |
                    (
                        self.data[col]
                        > upper
                    )
                ).sum()
            )

        self._update_report(
            "Outlier Validation",
            True,
            outlier_report
        )

        return True

    # =====================================================
    # CARDINALITY
    # =====================================================

    def validate_cardinality(self):

        report = {}

        for col in (
            self.config
            .high_cardinality_columns
        ):

            if col in self.data.columns:

                report[col] = int(
                    self.data[col]
                    .nunique()
                )

        self._update_report(
            "Cardinality Validation",
            True,
            report
        )

        return True

    # =====================================================
    # LEAKAGE CHECK
    # =====================================================

    def validate_data_leakage(self):

        found = []

        dataset_columns = [
            col.lower()
            for col in self.data.columns
        ]

        for leak_col in (
            self.config
            .leakage_columns
        ):

            if (
                leak_col.lower()
                in dataset_columns
            ):

                found.append(
                    leak_col
                )

        validation_passed = (
            len(found) == 0
        )

        self._update_report(
            "Leakage Validation",
            validation_passed,
            found
        )

        return validation_passed

    # =====================================================
    # CORRELATION ANALYSIS
    # =====================================================

    def validate_correlation(self):

        threshold = (
            self.config.thresholds
            .correlation_threshold
        )

        numeric_df = (
            self.data
            .select_dtypes(
                include=np.number
            )
        )

        corr_matrix = (
            numeric_df
            .corr()
            .abs()
        )

        upper = corr_matrix.where(
            np.triu(
                np.ones(
                    corr_matrix.shape
                ),
                k=1
            ).astype(bool)
        )

        high_corr_pairs = []

        for col in upper.columns:

            for row in upper.index:

                corr = upper.loc[
                    row,
                    col
                ]

                if (
                    pd.notnull(corr)
                    and
                    corr > threshold
                ):

                    high_corr_pairs.append(
                        {
                            "feature_1":
                                row,

                            "feature_2":
                                col,

                            "correlation":
                                round(
                                    float(
                                        corr
                                    ),
                                    4
                                )
                        }
                    )

        self._update_report(
            "Correlation Validation",
            True,
            high_corr_pairs
        )

        return True

    # =====================================================
    # DATA DRIFT
    # =====================================================

    def validate_data_drift(
        self,
        train_df,
        test_df
    ):

        threshold = (
            self.config.thresholds
            .drift_pvalue_threshold
        )

        drift_report = {}

        overall_pass = True

        numeric_cols = (
            train_df
            .select_dtypes(
                include=np.number
            )
            .columns
        )

        for col in numeric_cols:

            stat, p_value = ks_2samp(
                train_df[col]
                .dropna(),

                test_df[col]
                .dropna()
            )

            drift = (
                p_value
                <
                threshold
            )

            if drift:

                overall_pass = False

            drift_report[col] = {

                "ks_statistic":
                    float(stat),

                "p_value":
                    float(p_value),

                "drift_detected":
                    drift
            }

        self.drift_report = (
            drift_report
        )

        self._update_report(
            "Data Drift Validation",
            overall_pass,
            drift_report
        )

        return overall_pass

    # =====================================================
    # STATUS FILE
    # =====================================================

    def save_validation_status(
        self,
        validation_results
    ):

        overall_status = all(
            validation_results.values()
        )

        with open(
            self.config.STATUS_FILE,
            "w"
        ) as f:

            f.write(
                "DEMAND FORECASTING DATA VALIDATION\n"
            )

            f.write(
                "=" * 60
            )

            f.write("\n\n")

            for (
                name,
                status
            ) in validation_results.items():

                f.write(
                    f"{name}: "
                    f"{'PASSED' if status else 'FAILED'}\n"
                )

            f.write(
                "\n"
            )

            f.write(
                "=" * 60
            )

            f.write(
                "\nOVERALL STATUS: "
                f"{'PASSED' if overall_status else 'FAILED'}"
            )

    # =====================================================
    # MAIN PIPELINE
    # =====================================================

    def initiate_data_validation(self):

        logger.info(
            "Starting Data Validation"
        )

        self.generate_dataset_statistics()

        validation_results = {

            "Schema Validation":
                self.validate_schema(),

            "Datatype Validation":
                self.validate_datatypes(),

            "Missing Value Validation":
                self.validate_missing_values(),

            "Duplicate Validation":
                self.validate_duplicates(),

            "Numerical Range Validation":
                self.validate_numerical_ranges(),

            "Categorical Validation":
                self.validate_categorical_values(),

            "Target Validation":
                self.validate_target(),

            "Timestamp Validation":
                self.validate_timestamp(),

            "Time Series Continuity":
                self.validate_time_series_continuity(),

            "Business Rule Validation":
                self.validate_business_rules(),

            "Inventory Validation":
                self.validate_inventory_logic(),

            "Product Store Coverage":
                self.validate_product_store_coverage(),

            "Data Freshness":
                self.validate_data_freshness(),

            "Infinite Value Validation":
                self.validate_infinite_values(),

            "Outlier Validation":
                self.validate_outliers(),

            "Cardinality Validation":
                self.validate_cardinality(),

            "Leakage Validation":
                self.validate_data_leakage(),

            "Correlation Validation":
                self.validate_correlation()
        }

        self.save_validation_status(
            validation_results
        )

        self._save_reports()

        overall_status = all(
            validation_results.values()
        )

        logger.info(
            f"Validation Status: "
            f"{overall_status}"
        )

        return overall_status