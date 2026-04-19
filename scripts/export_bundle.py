"""训练完成后打包模型工件 → 写入 `{QS_EXPORT_ROOT}/{exp}/{run_id}/` → 可选 robocopy 到 vnpy 节点.

契约 1 (见 plan 文件): 每个 bundle 包含以下 5 个文件:

    params.pkl           qlib LGBModel (pickle)
    task.json            handler config + segments (JSON 可读)
    manifest.json        {core_version, core_git_sha, handler_module_path,
                          trained_at, bundle_version=1}
    requirements.lock    pip freeze
    baseline.parquet     训练末期特征分布 (PSI 基线, 见 qlib_strategy_core.baseline)

用法
----
    # 对实验里最新 run 打包
    python scripts/export_bundle.py --experiment rolling_exp

    # 指定 run_id
    python scripts/export_bundle.py --experiment rolling_exp --run-id 0068b7d99...

    # 推送到 vnpy 节点 (设置 VNPY_MODEL_ROOT env 变量)
    set VNPY_MODEL_ROOT=\\vnpy-node\share\models
    python scripts/export_bundle.py --experiment rolling_exp --push

手动兜底
-------
    打包后的目录可直接用网盘/U盘/RDP 拷贝到 vnpy 节点相应目录; vnpy 侧
    ModelRegistry 按 manifest.json 自动发现.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import pickle

# 兼容旧 MLflow 工件的 module_path (factor_factory.*). 脚本现在位于
# qlib_strategy_core/scripts/ (v0.4.1 从 qlib_strategy_dev/scripts/ 迁入).
# - core_root (scripts 的父目录) 上 sys.path 以便 import qlib_strategy_core 包本身
# - 若脚本位于 qlib_strategy_dev/vendor/qlib_strategy_core/scripts/ 的嵌套结构里,
#   则再上溯 3 层把 qlib_strategy_dev 根放进 sys.path, 让 factor_factory.*
#   shim 能被旧 pickle 通过 install_finder 解析到. 独立安装 core 时该路径
#   不存在, skip 即可.
_HERE = Path(__file__).resolve()
_CORE_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_CORE_ROOT))
if len(_HERE.parents) > 3:
    _DEV_ROOT = _HERE.parents[3]
    if (_DEV_ROOT / "factor_factory").exists():
        sys.path.insert(0, str(_DEV_ROOT))

from qlib_strategy_core._compat import install_finder

install_finder()

import qlib_strategy_core as core
from qlib_strategy_core.baseline import (
    compute_feature_distribution,
    BASELINE_SCHEMA_VERSION,
)
from qlib_strategy_core.config import QSConfig
from qlib_strategy_core.pipeline import RollingEnv, TaskBuilder

from qlib.workflow import R


BUNDLE_VERSION = 1


def _find_run(experiment_name: str, run_id: Optional[str]) -> Tuple[Any, Dict[str, Any]]:
    """Locate a recorder by experiment + optional run_id (default: latest)."""
    exp = R.get_exp(experiment_name=experiment_name)
    recorders = exp.list_recorders()
    if not recorders:
        raise RuntimeError(f"experiment '{experiment_name}' 下无 recorder")

    if run_id:
        if run_id not in recorders:
            raise RuntimeError(f"run_id '{run_id}' not in experiment '{experiment_name}'")
        rec = recorders[run_id]
        task = rec.load_object("task")
        return rec, task

    latest_rec = None
    latest_task: Optional[Dict[str, Any]] = None
    latest_end: Optional[pd.Timestamp] = None
    for _, rec in recorders.items():
        try:
            t = rec.load_object("task")
            te = t["dataset"]["kwargs"]["segments"]["test"][1]
        except Exception:
            continue
        ts = pd.Timestamp(te) if te is not None else pd.Timestamp.max
        if latest_end is None or ts > latest_end:
            latest_rec, latest_task, latest_end = rec, t, ts
    if latest_rec is None:
        raise RuntimeError("未找到可用 recorder")
    return latest_rec, latest_task


def _serialize_task_for_json(task: Dict[str, Any]) -> Dict[str, Any]:
    """task dict may contain pd.Timestamp / tuple — convert to JSON-safe."""
    def _conv(v):
        if isinstance(v, pd.Timestamp):
            return v.isoformat()
        if isinstance(v, dict):
            return {k: _conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_conv(x) for x in v]
        return v

    return _conv(copy.deepcopy(task))


def _detect_git_sha() -> Optional[str]:
    """Return short SHA of the qlib_strategy_core submodule, or None on failure."""
    try:
        core_dir = Path(core.__file__).resolve().parent.parent
        result = subprocess.run(
            ["git", "-C", str(core_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _build_baseline(task: Dict[str, Any], last_n_days: int = 60) -> Optional[pd.DataFrame]:
    """Rebuild DatasetH and compute a feature distribution over the last N days of the
    train segment. Returns None on failure (caller should continue — baseline is
    optional in bundle_version=1)."""
    try:
        train_seg = task["dataset"]["kwargs"]["segments"].get("train")
        if not train_seg:
            return None
        train_end = pd.Timestamp(train_seg[1])
        sample_start = train_end - pd.Timedelta(days=last_n_days)

        t = copy.deepcopy(task)
        t["dataset"]["kwargs"]["segments"] = {
            "test": (sample_start, train_end),
        }
        TaskBuilder.sync_handler_time_range(t)
        t["dataset"]["kwargs"]["handler"]["kwargs"]["use_cache"] = False

        dataset = TaskBuilder.build_dataset_from_task(t)
        # qlib DatasetH: prepare("test", col_set="feature") returns feature df
        features = dataset.prepare(["test"], col_set="feature")[0]
        if isinstance(features.columns, pd.MultiIndex):
            features.columns = [c[-1] if isinstance(c, tuple) else c for c in features.columns]
        return compute_feature_distribution(features, n_bins=20)
    except Exception as exc:
        print(f"[export_bundle] baseline 生成失败 (可跳过): {exc}", file=sys.stderr)
        return None


def _write_bundle(
    target_dir: Path,
    recorder: Any,
    task: Dict[str, Any],
    baseline_df: Optional[pd.DataFrame],
    experiment_name: str,
) -> None:
    """Write 5 artifacts into target_dir (atomic via .staging rename pattern)."""
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    staging = target_dir.with_suffix(".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # 1. params.pkl — reuse qlib recorder's artifact
    model = recorder.load_object("params.pkl")
    with open(staging / "params.pkl", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 2. task.json — human-readable frozen copy
    with open(staging / "task.json", "w", encoding="utf-8") as f:
        json.dump(_serialize_task_for_json(task), f, ensure_ascii=False, indent=2)

    # 3. manifest.json — version + SHA
    handler_cfg = task["dataset"]["kwargs"]["handler"]
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "core_version": core.__version__,
        "core_git_sha": _detect_git_sha(),
        "python_version": sys.version.split()[0],
        "qlib_version": _get_qlib_version(),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "handler_class": handler_cfg.get("class"),
        "handler_module_path": handler_cfg.get("module_path"),
        "experiment_name": experiment_name,
        "run_id": _extract_run_id(recorder),
        "baseline_present": baseline_df is not None,
        "baseline_schema_version": BASELINE_SCHEMA_VERSION if baseline_df is not None else None,
    }
    with open(staging / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 4. requirements.lock
    req_output = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    (staging / "requirements.lock").write_text(req_output.stdout, encoding="utf-8")

    # 5. baseline.parquet (optional)
    if baseline_df is not None and not baseline_df.empty:
        baseline_df.to_parquet(staging / "baseline.parquet", index=False)

    # atomic swap
    if target_dir.exists():
        shutil.rmtree(target_dir)
    staging.rename(target_dir)


def _get_qlib_version() -> str:
    try:
        import qlib
        return getattr(qlib, "__version__", "unknown")
    except Exception:
        return "unknown"


def _extract_run_id(recorder: Any) -> str:
    if hasattr(recorder, "id") and recorder.id:
        return str(recorder.id)
    if hasattr(recorder, "info") and isinstance(recorder.info, dict):
        return str(recorder.info.get("id", ""))
    return str(recorder)


def _robocopy(src: Path, dst: Path) -> int:
    """Windows-native mirror copy. Returns 0/1/2/3 = success per robocopy convention."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Use /MIR to mirror + /XO to skip older destination files (safe re-run)
    cmd = ["robocopy", str(src), str(dst), "/E", "/R:3", "/W:5", "/NFL", "/NDL"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # robocopy exit codes 0-3 are all successful (no errors)
    if result.returncode < 8:
        return 0
    print(f"[export_bundle] robocopy 失败 ({result.returncode}): {result.stdout[-300:]}", file=sys.stderr)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="训练工件打包 + 可选推送到 vnpy 节点")
    parser.add_argument("--experiment", default=None, help="MLflow experiment name (default: from QSConfig)")
    parser.add_argument("--run-id", default=None, help="specific run_id; default = latest by test-end")
    parser.add_argument("--export-root", default=None, help="bundle staging dir (default: QS_EXPORT_ROOT or ./qs_exports)")
    parser.add_argument("--push", action="store_true", help="robocopy bundle to VNPY_MODEL_ROOT")
    parser.add_argument("--vnpy-root", default=None, help="vnpy model root (default: VNPY_MODEL_ROOT env)")
    parser.add_argument("--baseline-days", type=int, default=60, help="days from train end to sample for baseline")
    args = parser.parse_args()

    cfg = QSConfig.from_env()
    RollingEnv.setup_env()
    RollingEnv.init_qlib(
        provider_uri=cfg.provider_uri,
        mlflow_tracking_uri=cfg.mlflow_tracking_uri,
        mlflow_path=cfg.mlflow_path,
    )

    experiment_name = args.experiment or cfg.experiment_name
    export_root = Path(
        args.export_root or os.getenv("QS_EXPORT_ROOT") or (Path.cwd() / "qs_exports")
    )

    recorder, task = _find_run(experiment_name, args.run_id)
    run_id = _extract_run_id(recorder)
    target_dir = export_root / experiment_name / run_id

    print(f"[export_bundle] packaging {experiment_name}/{run_id} → {target_dir}")
    baseline = _build_baseline(task, last_n_days=args.baseline_days)
    _write_bundle(target_dir, recorder, task, baseline, experiment_name)
    print(f"[export_bundle] bundle complete: {target_dir}")

    if args.push:
        vnpy_root = args.vnpy_root or os.getenv("VNPY_MODEL_ROOT")
        if not vnpy_root:
            print("[export_bundle] --push 需要 --vnpy-root 或 VNPY_MODEL_ROOT 环境变量", file=sys.stderr)
            return 1
        push_target = Path(vnpy_root) / experiment_name / run_id
        print(f"[export_bundle] pushing → {push_target}")
        rc = _robocopy(target_dir, push_target)
        if rc != 0:
            return rc
        print(f"[export_bundle] pushed OK")

    return 0


if __name__ == "__main__":
    sys.exit(main())
