"""Inference API — consumed by both the training-side verification CLI and
vnpy's ``QlibPredictor``.

The canonical flow:

1. Find a recorder in an MLflow experiment (latest by default, or explicit run_id).
2. Load the frozen ``task`` artifact + trained ``params.pkl``.
3. Override ``segments["test"]`` to a live window ``[live_end - lookback, live_end]``.
4. Sync handler time fields via ``TaskBuilder.sync_handler_time_range``.
5. Rebuild the DatasetH via ``TaskBuilder.build_dataset_from_task`` (feature
   construction is identical to training because the task dict is identical).
6. Call ``model.predict(dataset, segment="test")`` → pred_df.

Callers wrap this function with their own pre/post logic. This module owns
nothing except the predict step itself.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from qlib.workflow import R

from qlib_strategy_core.pipeline import TaskBuilder


LIVE_HANDLER_DEFAULTS: Dict[str, Any] = {
    "use_cache": False,
}


def _find_latest_recorder(experiment_name: str) -> Tuple[Any, Dict[str, Any]]:
    """Pick the recorder with the latest ``test`` segment end date."""
    exp = R.get_exp(experiment_name=experiment_name)
    recorders = exp.list_recorders()
    if not recorders:
        raise RuntimeError(f"experiment '{experiment_name}' 下没有任何 recorder")

    latest_rec = None
    latest_task: Optional[Dict[str, Any]] = None
    latest_end: Optional[pd.Timestamp] = None

    for _, rec in recorders.items():
        try:
            task = rec.load_object("task")
            test_end = task["dataset"]["kwargs"]["segments"]["test"][1]
        except Exception:
            continue
        test_end_ts = pd.Timestamp(test_end) if test_end is not None else pd.Timestamp.max
        if latest_end is None or test_end_ts > latest_end:
            latest_rec = rec
            latest_task = task
            latest_end = test_end_ts

    if latest_rec is None or latest_task is None:
        raise RuntimeError("未能从任何 recorder 中解析出 test 段结束时间")
    return latest_rec, latest_task


def predict_from_recorder(
    experiment_name: str,
    live_end: pd.Timestamp,
    lookback_days: int = 250,
    handler_overrides: Optional[Dict[str, Any]] = None,
    recorder: Any = None,
    task: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run daily inference using the latest recorder in ``experiment_name``.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name. Ignored if ``recorder`` is provided.
    live_end : Timestamp
        Inference window end date (inclusive).
    lookback_days : int
        Natural days to look back for feature construction. The handler needs
        enough history to compute lagged factors (e.g., 60-day EMAs).
    handler_overrides : dict, optional
        Overrides merged into ``task["dataset"]["kwargs"]["handler"]["kwargs"]``.
        Priority: ``handler_overrides`` > ``LIVE_HANDLER_DEFAULTS`` > original task.
        ``LIVE_HANDLER_DEFAULTS`` disables the factor cache by default.
    recorder, task : optional
        If both are provided, skip the MLflow lookup. Useful when vnpy side
        has already loaded these from a bundle directory.

    Returns
    -------
    (pred_df, task)
        * pred_df: MultiIndex (datetime, instrument) DataFrame with ``score`` column.
        * task: deepcopy of the task dict with overrides applied, for downstream
          consumers (e.g., extract strategy topk/n_drop from ``task["record"]``).
    """
    if recorder is None or task is None:
        recorder, task = _find_latest_recorder(experiment_name)

    task = copy.deepcopy(task)
    live_start = live_end - pd.Timedelta(days=lookback_days)
    task["dataset"]["kwargs"]["segments"] = {
        "test": (live_start, live_end),
    }
    TaskBuilder.sync_handler_time_range(task)

    merged_overrides = dict(LIVE_HANDLER_DEFAULTS)
    if handler_overrides:
        merged_overrides.update(handler_overrides)
    handler_kwargs = task["dataset"]["kwargs"]["handler"]["kwargs"]
    changed = {k: v for k, v in merged_overrides.items() if handler_kwargs.get(k) != v}
    handler_kwargs.update(merged_overrides)
    if changed:
        print(f"[predict_from_recorder] handler 覆盖: {changed}")

    dataset = TaskBuilder.build_dataset_from_task(task)
    model = recorder.load_object("params.pkl")

    pred = model.predict(dataset, segment="test")
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    return pred, task
