# qlib_strategy_core

Shared runtime library for `qlib_strategy_dev` (training) and `vnpy_strategy_dev` (inference).

Contains:

- `handlers/` — qlib DataHandlerLP factor definitions (Alpha158Custom + friends)
- `alpha_factor_store.py` — Hive-partitioned Parquet factor store (build/load)
- `pipeline.py` — `RollingEnv`, `TaskBuilder` helpers (qlib + MLflow init, handler time sync, dataset rebuild)
- `inference.py` — `predict_from_recorder` for daily live prediction
- `baseline.py` — `compute_feature_distribution` for PSI baseline parquet export
- `dashboard_contract.py` — pydantic payload schemas for training → mlearnweb HTTP interface
- `config.py` — `QSConfig.from_env()` for cross-host path configuration
- `_compat/legacy_paths.py` — MetaPathFinder for legacy `factor_factory.alphas.*` module paths

Consumers install this package via `pip install -e ./qlib_strategy_core` (submodule development)
or `pip install git+ssh://git@github.com/xiaospica/qlib_strategy_core.git@v0.1.0` (production).

## Versioning

Follow SemVer. Major version bump signals incompatibility (e.g., handler class path change or
payload `schema_version` bump); consumers must explicitly opt into these.
