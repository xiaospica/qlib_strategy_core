"""Cross-host path configuration via environment variables.

``QSConfig.from_env()`` provides a single read-once dataclass snapshot.
All paths default to the training box's current Windows layout for
backward compat; override via environment on the inference box.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


_DEFAULT_BASE = r"F:\Quant\code\qlib_strategy_dev"
_DEFAULT_PROVIDER_URI = rf"{_DEFAULT_BASE}\factor_factory\qlib_data_bin"
_DEFAULT_MLFLOW_PATH = rf"{_DEFAULT_BASE}\mlruns"
_DEFAULT_MLFLOW_URI = f"file:///{_DEFAULT_MLFLOW_PATH.replace(os.sep, '/')}"
_DEFAULT_FACTOR_STORE = rf"{_DEFAULT_BASE}\factor_factory\.cache\factor_store"
_DEFAULT_FILTER_PARQUET = rf"{_DEFAULT_BASE}\factor_factory\csi300_custom_filtered.parquet"


@dataclass(frozen=True)
class QSConfig:
    provider_uri: str
    mlflow_tracking_uri: str
    mlflow_path: str
    experiment_name: str
    factor_store_path: str
    filter_parquet_path: str
    output_dir: str
    mode: str  # "train" | "inference"

    @classmethod
    def from_env(cls) -> "QSConfig":
        return cls(
            provider_uri=os.getenv("QS_PROVIDER_URI", _DEFAULT_PROVIDER_URI),
            mlflow_tracking_uri=os.getenv("QS_MLFLOW_TRACKING_URI", _DEFAULT_MLFLOW_URI),
            mlflow_path=os.getenv("QS_MLFLOW_PATH", _DEFAULT_MLFLOW_PATH),
            experiment_name=os.getenv("QS_EXPERIMENT", "rolling_exp"),
            factor_store_path=os.getenv("QS_FACTOR_STORE_PATH", _DEFAULT_FACTOR_STORE),
            filter_parquet_path=os.getenv("QS_FILTER_PARQUET", _DEFAULT_FILTER_PARQUET),
            output_dir=os.getenv("QS_OUTPUT_DIR", os.path.join(os.getcwd(), "output")),
            mode=os.getenv("QS_MODE", "train"),
        )
