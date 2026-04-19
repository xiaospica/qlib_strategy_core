"""Shared pipeline helpers — used by both training and inference.

``RollingEnv`` owns the qlib + MLflow initialization.
``TaskBuilder`` owns the time-range sync + DatasetH construction from a task dict.

These are the two choke points for train/inference symmetry: if the inference
side rebuilds a DatasetH from the same task artifact and syncs handler times
the same way, the feature matrix is byte-identical to training.

MLflow-dependent imports (``qlib.workflow``, ``qlib.workflow.task.gen``) are
deferred so that inference-only environments (e.g., vnpy nodes without
``mlflow`` installed) can still import this module for the predict path.
"""

from __future__ import annotations

import copy
import os
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config


# ROLL_SD constant duplicated here to avoid a qlib.workflow.task.gen import at
# module load; RollingGen itself is lazily imported inside generate_rolling_tasks.
_DEFAULT_ROLLING_TYPE = "-"  # == RollingGen.ROLL_SD when imported lazily


class RollingEnv:
    """环境变量、警告、日志、qlib / mlflow 初始化. 幂等."""

    _env_ready = False
    _qlib_ready = False

    @staticmethod
    def setup_env() -> None:
        if RollingEnv._env_ready:
            return
        import logging
        import warnings

        warnings.filterwarnings("ignore")
        os.environ.setdefault("PYTHONWARNINGS", "ignore")
        os.environ.setdefault("LOKY_PICKLER", "pickle")
        os.environ.setdefault("JOBLIB_START_METHOD", "spawn")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        logging.basicConfig(level=logging.ERROR)
        RollingEnv._env_ready = True

    @staticmethod
    def init_qlib(
        provider_uri: str,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_path: Optional[str] = None,
        region: str = REG_CN,
    ) -> None:
        if mlflow_path:
            os.makedirs(mlflow_path, exist_ok=True)

        if mlflow_tracking_uri:
            import mlflow  # lazy: training side only

            mlflow.set_tracking_uri(mlflow_tracking_uri)

        qlib.init(provider_uri=provider_uri, region=region)

        if mlflow_tracking_uri:
            from qlib.workflow import R  # lazy: avoid mlflow dep on inference boxes

            R.set_uri(mlflow_tracking_uri)

        RollingEnv._qlib_ready = True


class TaskBuilder:
    """task dict 的构造与时间字段同步.

    ``sync_handler_time_range`` 和 ``build_dataset_from_task`` 是训推一体的
    核心助手: 训练路径在 ``generate_rolling_tasks`` 里调用; 实盘推理在覆写
    ``segments["test"]`` 之后也调这两个函数, 保证一致性.
    """

    @staticmethod
    def sync_handler_time_range(task: Dict[str, Any]) -> Dict[str, Any]:
        """把 handler 的四个时间字段同步到 dataset.segments 对应值."""
        handler_kwargs = task["dataset"]["kwargs"]["handler"]["kwargs"]
        segments = task["dataset"]["kwargs"]["segments"]

        train_seg = segments.get("train")
        if train_seg is not None:
            handler_kwargs["fit_start_time"] = train_seg[0]
            handler_kwargs["fit_end_time"] = train_seg[1]
            handler_kwargs["start_time"] = train_seg[0]

        test_seg = segments.get("test")
        if test_seg is not None:
            test_end = test_seg[1]
            if test_end is None:
                test_end = pd.Timestamp("2030-12-31")
            handler_kwargs["end_time"] = test_end

        return task

    @staticmethod
    def build_dataset_from_task(task: Dict[str, Any]):
        """按 task['dataset'] 构造 DatasetH. 训推一体共用."""
        return init_instance_by_config(task["dataset"])

    @staticmethod
    def generate_rolling_tasks(
        base_task_config: Any,
        rolling_step: int = 500,
        rolling_type: Optional[str] = None,
        custom_segments: Optional[List[Dict[str, Tuple]]] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """基于 base task + RollingGen 或自定义 segments 生成若干滚动 task, 并同步 handler 时间.

        Training-side only (depends on qlib.workflow.task.gen).
        """
        from qlib.workflow.task.gen import RollingGen, task_generator  # lazy

        if rolling_type is None:
            rolling_type = RollingGen.ROLL_SD

        print("========== task_generating ==========")
        if custom_segments:
            base = base_task_config if isinstance(base_task_config, dict) else base_task_config[0]
            tasks = []
            for seg in custom_segments:
                t = copy.deepcopy(base)
                t["dataset"]["kwargs"]["segments"] = seg
                tasks.append(t)
        else:
            rolling_gen = RollingGen(step=rolling_step, rtype=rolling_type)
            tasks = task_generator(tasks=base_task_config, generators=rolling_gen)
        for task in tasks:
            TaskBuilder.sync_handler_time_range(task)
        if verbose:
            pprint(tasks)
        return tasks
