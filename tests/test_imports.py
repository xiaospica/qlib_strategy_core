"""Smoke tests — ensure every public module imports without error."""


def test_root_version():
    import qlib_strategy_core

    assert qlib_strategy_core.__version__ == "0.1.0"


def test_handlers_import():
    from qlib_strategy_core.handlers import Alpha158Custom, Alpha158CustomDataLoader

    assert Alpha158Custom is not None
    assert Alpha158CustomDataLoader is not None


def test_pipeline_import():
    from qlib_strategy_core.pipeline import RollingEnv, TaskBuilder

    assert hasattr(RollingEnv, "init_qlib")
    assert hasattr(TaskBuilder, "sync_handler_time_range")
    assert hasattr(TaskBuilder, "build_dataset_from_task")


def test_inference_import():
    from qlib_strategy_core.inference import predict_from_recorder, LIVE_HANDLER_DEFAULTS

    assert callable(predict_from_recorder)
    assert LIVE_HANDLER_DEFAULTS["use_cache"] is False


def test_config_defaults():
    from qlib_strategy_core.config import QSConfig

    cfg = QSConfig.from_env()
    assert cfg.experiment_name
    assert cfg.mode in ("train", "inference")


def test_dashboard_contract_schema_version():
    from qlib_strategy_core.dashboard_contract import (
        TrainingRecordPayload,
        RunMappingPayload,
        SCHEMA_VERSION,
    )

    p = TrainingRecordPayload(name="x", experiment_name="y")
    assert p.schema_version == SCHEMA_VERSION


def test_baseline_schema_constants():
    from qlib_strategy_core.baseline import (
        COL_FEATURE, COL_BIN_ID, COL_EDGE_LO, COL_EDGE_HI, COL_PROBABILITY,
        BASELINE_SCHEMA_VERSION,
    )

    assert BASELINE_SCHEMA_VERSION == 1
    assert COL_FEATURE == "feature"


def test_legacy_path_finder_install():
    from qlib_strategy_core._compat.legacy_paths import install_finder, LEGACY_TO_CURRENT

    install_finder()
    install_finder()  # idempotent

    # Verify the alias resolves at import time
    import importlib
    mod = importlib.import_module("factor_factory.alpha_factor_store")
    assert mod is not None
    assert hasattr(mod, "load_factor_slice")
