from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    unzip_data_dir: Path
    STATUS_FILE: Path
    REPORT_FILE: Path
    DRIFT_REPORT_FILE: Path
    STATS_REPORT_FILE: Path
    all_schema: dict
    numerical_ranges: dict
    categorical_values: dict
    thresholds: dict
    high_cardinality_columns: list
    leakage_columns: list
    timestamp_column: str
    target_column: str