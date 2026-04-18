"""Inference API — shared between training-side verification CLI and vnpy's
``QlibPredictor``.

Two entry points:

* :func:`predict_from_recorder` — accepts an MLflow ``experiment_name``, does
  recorder lookup, loads ``task`` + ``params.pkl`` artifacts via qlib's
  ``recorder.load_object``. Requires ``mlflow`` (imported lazily).
  Used on training machines.

* :func:`predict_from_bundle` — accepts a filesystem directory containing a
  Phase-1 bundle (``params.pkl`` + ``task.json``). Pure ``pickle`` + ``json``
  load, no MLflow / no qlib.workflow dependency.
  Used on inference machines (e.g., vnpy node) that only hold a rsync'd bundle
  and may not have mlflow installed.

Both funnel into :func:`_predict_core`, which is the only place model inference
actually runs. Callers are responsible for any pre/post downstream formatting
(TopK selection, order generation, metrics).
"""

from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from qlib_strategy_core.pipeline import TaskBuilder


LIVE_HANDLER_DEFAULTS: Dict[str, Any] = {
    "use_cache": False,
}


# ---------------------------------------------------------------------------
# Internal core
# ---------------------------------------------------------------------------


def _predict_core(
    model: Any,
    task: Dict[str, Any],
    live_end: pd.Timestamp,
    lookback_days: int = 250,
    handler_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """The only place model.predict is actually invoked.

    Caller supplies a loaded model object + the task dict; this function handles
    the segment override, handler time sync, dataset rebuild, and predict call.
    """
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
        print(f"[qlib_strategy_core] handler 覆盖: {changed}")

    dataset = TaskBuilder.build_dataset_from_task(task)
    pred = model.predict(dataset, segment="test")
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    return pred, task


# ---------------------------------------------------------------------------
# Training-side entry: lookup via MLflow experiment name
# ---------------------------------------------------------------------------


def _find_latest_recorder(experiment_name: str) -> Tuple[Any, Dict[str, Any]]:
    """Pick the recorder with the latest ``test`` segment end date."""
    from qlib.workflow import R  # lazy: avoid qlib.workflow dep on inference boxes

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
    """Training-side inference — reads model + task from MLflow.

    If ``recorder`` and ``task`` are both provided, skip MLflow lookup and use
    them directly (still uses ``recorder.load_object("params.pkl")``).
    """
    if recorder is None or task is None:
        recorder, task = _find_latest_recorder(experiment_name)

    model = recorder.load_object("params.pkl")
    return _predict_core(
        model=model,
        task=task,
        live_end=live_end,
        lookback_days=lookback_days,
        handler_overrides=handler_overrides,
    )


# ---------------------------------------------------------------------------
# Inference-side entry: load from a rsync'd bundle directory (no MLflow)
# ---------------------------------------------------------------------------


def _task_from_json_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Read task.json and re-hydrate Timestamps / tuples from JSON primitives."""
    with open(path, "r", encoding="utf-8") as f:
        task = json.load(f)

    # segments: lists in JSON → tuples in Python, strings → Timestamps
    segments = task.get("dataset", {}).get("kwargs", {}).get("segments")
    if isinstance(segments, dict):
        for k, v in list(segments.items()):
            if isinstance(v, list) and len(v) == 2:
                segments[k] = (
                    pd.Timestamp(v[0]) if v[0] is not None else None,
                    pd.Timestamp(v[1]) if v[1] is not None else None,
                )
    return task


def predict_from_bundle(
    bundle_dir: Union[str, Path],
    live_end: pd.Timestamp,
    lookback_days: int = 250,
    handler_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Inference-side entry — loads model from a Phase-1 bundle directory.

    Parameters
    ----------
    bundle_dir : str | Path
        Directory containing ``params.pkl`` + ``task.json`` (output of
        ``scripts/export_bundle.py``). May also contain ``manifest.json``
        (for version checks) and ``baseline.parquet`` (for PSI).
    live_end, lookback_days, handler_overrides
        Same semantics as :func:`predict_from_recorder`.

    Returns
    -------
    (pred_df, task) — identical shape to :func:`predict_from_recorder`.

    Raises
    ------
    FileNotFoundError : if ``params.pkl`` or ``task.json`` missing.
    """
    bundle_dir = Path(bundle_dir)
    params_path = bundle_dir / "params.pkl"
    task_path = bundle_dir / "task.json"

    if not params_path.exists():
        raise FileNotFoundError(f"bundle missing params.pkl: {params_path}")
    if not task_path.exists():
        raise FileNotFoundError(f"bundle missing task.json: {task_path}")

    with open(params_path, "rb") as f:
        model = pickle.load(f)
    task = _task_from_json_file(task_path)

    return _predict_core(
        model=model,
        task=task,
        live_end=live_end,
        lookback_days=lookback_days,
        handler_overrides=handler_overrides,
    )
