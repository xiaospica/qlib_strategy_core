"""Subprocess inference entry point — invoked by vnpy main process.

Invoked as::

    python -m qlib_strategy_core.cli.run_inference \
        --bundle-dir  /path/to/model/run_id \
        --live-end    2026-04-21 \
        --lookback    60 \
        --out-dir     /path/to/output/csi300/20260421 \
        --strategy    csi300_lgb \
        --provider-uri  F:/Quant/.../qlib_data_bin

Output (written to ``--out-dir`` in this order, each atomically via ``.tmp``
rename):

1. ``predictions.parquet`` — MultiIndex (datetime, instrument), column score
2. ``metrics.json`` — IC / RankIC / PSI / KS / histogram / feat_missing / model_run_id
3. ``diagnostics.json`` — SENTINEL. Its existence means "integration complete".

Exit codes:

* 0 — completed (check ``status`` field in diagnostics for ok / empty)
* 1 — Python exception before diagnostics could be written
* 2 — invalid args (argparse handles)

Callers should:

* Invoke with ``timeout`` to guard against OOM/hangs
* Check ``diagnostics.json.schema_version`` matches their expected version
* Treat missing ``diagnostics.json`` as failure regardless of exit code
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Ensure the vendored qlib at ../../qlib is preferred over any pip-installed one.
# Layout: vendor/qlib_strategy_core/qlib_strategy_core/cli/run_inference.py
#                ^^^^^^^^^^^^^^^^^^^ this dir is what we need on sys.path
_CORE_ROOT = Path(__file__).resolve().parents[2]
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

import pandas as pd


DIAGNOSTICS_SCHEMA_VERSION = 1


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def _atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="qlib_strategy_core subprocess inference entry — daily pipeline",
    )
    p.add_argument("--bundle-dir", required=True, help="Directory with params.pkl + task.json")
    p.add_argument(
        "--live-end",
        required=False,
        default=None,
        help="单日推理：Inference end date YYYY-MM-DD。与 --live-end-range 互斥",
    )
    p.add_argument(
        "--live-end-range",
        required=False,
        default=None,
        help=(
            "批量推理（Phase 4 加速回放）：start,end 形式如 2026-01-01,2026-04-29。"
            "一次性 init qlib + load bundle + build dataset + predict，按日切片输出。"
            "out-dir 解释为父目录，子进程内部按 yyyymmdd 建子目录。互斥 --live-end。"
        ),
    )
    p.add_argument("--lookback", type=int, default=60, help="Natural days lookback")
    p.add_argument("--out-dir", required=True, help="Output directory for three-file bundle")
    p.add_argument("--strategy", default="default", help="Strategy name (metadata)")
    p.add_argument("--provider-uri", default=None, help="qlib data provider URI; inferred from env if omitted")
    p.add_argument(
        "--baseline",
        default=None,
        help="baseline.parquet path for PSI. Defaults to {bundle-dir}/baseline.parquet",
    )
    p.add_argument(
        "--install-legacy-path",
        action="store_true",
        help="Install MetaPathFinder for pre-0.2 MLflow artifact module paths (factor_factory.*)",
    )
    p.add_argument(
        "--filter-parquet",
        default=None,
        help=(
            "Override handler's filter_parquet_path at runtime. "
            "Live inference should pass snapshots/filtered/csi300_filtered_{live_end}.parquet "
            "so the universe filter reflects the exact frozen snapshot for that date. "
            "If omitted, handler kwargs from task.json are used as-is."
        ),
    )
    return p


def _load_manifest(bundle_dir: Path) -> Dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _run_batch_mode(args, t0: float) -> int:
    """Phase 4 批量推理模式：一次性 init qlib + load bundle + predict 整个区间，按日切片输出。"""
    from qlib_strategy_core import __version__ as core_version
    from qlib_strategy_core.inference import predict_range_from_bundle
    from qlib_strategy_core.pipeline import RollingEnv

    if args.install_legacy_path:
        from qlib_strategy_core._compat import install_finder
        install_finder()

    bundle_dir = Path(args.bundle_dir)
    if not bundle_dir.exists():
        raise FileNotFoundError(f"bundle dir not found: {bundle_dir}")

    manifest = _load_manifest(bundle_dir)
    model_run_id = manifest.get("run_id") or bundle_dir.name

    provider_uri = args.provider_uri or os.getenv("QLIB_PROVIDER_URI") or os.getenv("QS_PROVIDER_URI")
    if not provider_uri:
        raise RuntimeError("provider_uri required (--provider-uri or QLIB_PROVIDER_URI env)")

    parts = args.live_end_range.split(",")
    if len(parts) != 2:
        raise ValueError(f"--live-end-range 必须是 start,end 形式: {args.live_end_range!r}")
    range_start = pd.Timestamp(parts[0].strip()).normalize()
    range_end = pd.Timestamp(parts[1].strip()).normalize()
    if range_start > range_end:
        raise ValueError(f"start {range_start} > end {range_end}")

    # qlib + handler 一次性加载（这是 ~80% 的回放开销，批量模式核心收益）
    RollingEnv.setup_env()
    RollingEnv.init_qlib(provider_uri=provider_uri)

    handler_overrides: Dict[str, Any] = {}
    if args.filter_parquet:
        handler_overrides["filter_parquet"] = args.filter_parquet

    # segment 覆盖整个回放窗口 + lookback 前缀（让首日也有完整因子计算窗口）
    segment_start = range_start - pd.Timedelta(days=args.lookback)
    print(
        f"[run_inference batch] strategy={args.strategy} range=[{range_start.date()}, "
        f"{range_end.date()}] lookback={args.lookback} segment=[{segment_start.date()}, "
        f"{range_end.date()}]"
    )

    # 一次 predict 出整段
    pred_df_full, _task = predict_range_from_bundle(
        bundle_dir=bundle_dir,
        segment_start=segment_start,
        segment_end=range_end,
        handler_overrides=handler_overrides or None,
    )

    # datetime level 兼容（与单日模式相同处理）
    dt_idx = pd.to_datetime(pred_df_full.index.get_level_values("datetime"))

    # 按日切片写每日子目录
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    n_days_with_data = 0
    n_days_total = 0

    # 遍历整段交易日
    cursor = range_start
    while cursor <= range_end:
        n_days_total += 1
        day_str = cursor.strftime("%Y%m%d")
        day_iso = cursor.strftime("%Y-%m-%d")
        day_norm = cursor.normalize()
        day_out = out_root / day_str
        day_out.mkdir(parents=True, exist_ok=True)

        day_t0 = time.time()
        pred_df_today = pred_df_full[dt_idx == day_norm]

        # 每日 diagnostics + predictions（不写 metrics.json — 简化版）
        diag_day: Dict[str, Any] = {
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "strategy": args.strategy,
            "live_end": day_iso,
            "lookback_days": args.lookback,
            "started_at": datetime.fromtimestamp(day_t0).isoformat(timespec="seconds"),
            "model_run_id": model_run_id,
            "core_version": core_version,
            "batch_mode": True,
        }

        if len(pred_df_today) > 0:
            _atomic_write_parquet(day_out / "predictions.parquet", pred_df_today)
            diag_day.update({
                "status": "ok",
                "exit_code": 0,
                "duration_ms": int((time.time() - day_t0) * 1000),
                "rows": int(len(pred_df_today)),
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            })
            n_days_with_data += 1
        else:
            diag_day.update({
                "status": "empty",
                "exit_code": 0,
                "duration_ms": int((time.time() - day_t0) * 1000),
                "rows": 0,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            })

        _atomic_write_json(day_out / "diagnostics.json", diag_day)
        cursor += pd.Timedelta(days=1)

    total_ms = int((time.time() - t0) * 1000)
    print(
        f"[run_inference batch] done {n_days_with_data}/{n_days_total} days have data, "
        f"total {total_ms}ms (= {total_ms / n_days_total:.0f}ms/day average)"
    )
    return 0


def main() -> int:
    t0 = time.time()
    args = _build_arg_parser().parse_args()

    # Phase 4 加速回放：批量模式分支（互斥单日模式）
    if args.live_end_range and args.live_end:
        print("[run_inference] ERROR: --live-end 与 --live-end-range 互斥", file=sys.stderr)
        return 2
    if args.live_end_range:
        try:
            return _run_batch_mode(args, t0)
        except Exception as exc:
            traceback.print_exc()
            print(f"[run_inference batch] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    if not args.live_end:
        print("[run_inference] ERROR: 需要 --live-end 或 --live-end-range 之一", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diag_path = out_dir / "diagnostics.json"
    diag_base: Dict[str, Any] = {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "strategy": args.strategy,
        "live_end": args.live_end,
        "lookback_days": args.lookback,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }

    # Optional legacy path compat (old MLflow artifacts may pickle
    # module_path = "factor_factory.alphas.alpha_158_custom_qlib")
    if args.install_legacy_path:
        from qlib_strategy_core._compat import install_finder
        install_finder()

    try:
        from qlib_strategy_core import __version__ as core_version
        from qlib_strategy_core.inference import predict_from_bundle
        from qlib_strategy_core.pipeline import RollingEnv
        from qlib_strategy_core.metrics import (
            compute_ic,
            compute_rank_ic,
            compute_psi_by_feature,
            compute_ks_by_feature,
            compute_prediction_stats,
            compute_score_histogram,
            compute_feature_missing_rate,
            summarize_psi,
        )

        bundle_dir = Path(args.bundle_dir)
        if not bundle_dir.exists():
            raise FileNotFoundError(f"bundle dir not found: {bundle_dir}")

        manifest = _load_manifest(bundle_dir)
        model_run_id = manifest.get("run_id") or bundle_dir.name

        provider_uri = args.provider_uri or os.getenv("QLIB_PROVIDER_URI") or os.getenv("QS_PROVIDER_URI")
        if not provider_uri:
            raise RuntimeError("provider_uri required (--provider-uri or QLIB_PROVIDER_URI env)")

        RollingEnv.setup_env()
        RollingEnv.init_qlib(provider_uri=provider_uri)

        live_end_ts = pd.Timestamp(args.live_end).normalize()

        # Phase 4 v2: 按 live_end 覆盖 handler 的 filter_parquet. 注意 handler
        # 的参数名是 ``filter_parquet`` (见 Alpha158CustomDataLoader.__init__),
        # 不是 filter_parquet_path.
        handler_overrides: Dict[str, Any] = {}
        if args.filter_parquet:
            handler_overrides["filter_parquet"] = args.filter_parquet

        # 1. predict
        pred_df, task = predict_from_bundle(
            bundle_dir=bundle_dir,
            live_end=live_end_ts,
            lookback_days=args.lookback,
            handler_overrides=handler_overrides or None,
        )

        # pred_df 从 DatasetH test segment = [live_end - lookback, live_end] 来,
        # 在 qlib bin 每日重建(末尾=live_end)的场景下, 每跨一天就多一天 pred
        # (见 smoke 实测: 20260407 一天 → 20260417 九天, n_pred 300→2700 累加).
        # 落盘的 predictions.parquet 和 n_predictions 必须只反映 live_end 当日
        # 预测(CSI300 ≈ 300 行), 否则下游 selections/topk/metrics 会被历史行
        # 污染. IC 计算仍用完整 pred_df (历史部分的 label 可能可算, 保留机会).
        live_end_day = pd.Timestamp(args.live_end).normalize()
        # qlib 数据 index 的 datetime level 在某些版本/数据源下可能是普通 Index（object/string dtype），
        # 而非 DatetimeIndex；先 pd.to_datetime 强制转换以兼容（Index.normalize 仅 DatetimeIndex 支持）。
        dt_idx = pd.to_datetime(pred_df.index.get_level_values("datetime"))
        pred_df_today = pred_df[dt_idx.normalize() == live_end_day]

        _atomic_write_parquet(out_dir / "predictions.parquet", pred_df_today)

        # 2. metrics
        metrics: Dict[str, Any] = {
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "strategy": args.strategy,
            "trade_date": args.live_end,
            "model_run_id": model_run_id,
            "core_version": core_version,
            "n_predictions": int(len(pred_df_today)),
        }

        metrics.update(compute_prediction_stats(pred_df_today))
        metrics["score_histogram"] = compute_score_histogram(pred_df_today, n_bins=20)

        # Build test-segment dataset once — used for IC (labels) + PSI/KS (features)
        dataset = None
        try:
            from qlib_strategy_core.pipeline import TaskBuilder
            dataset = TaskBuilder.build_dataset_from_task(task)
        except Exception as exc:
            metrics["dataset_error"] = f"{type(exc).__name__}: {exc}"

        # IC / RankIC — latest-day cross-section vs. realized forward returns.
        # NaN (e.g. when the most recent day's forward window exceeds available
        # bin data) is emitted as JSON null; downstream `metrics.get("ic")` → None.
        import math
        metrics["ic"] = None
        metrics["rank_ic"] = None
        if dataset is not None:
            try:
                label_prep = dataset.prepare(["test"], col_set="label")[0]
                label_series = (
                    label_prep.iloc[:, 0]
                    if isinstance(label_prep, pd.DataFrame) and label_prep.shape[1] >= 1
                    else label_prep
                )
                if hasattr(label_series, "dropna"):
                    label_series = label_series.dropna()
                if len(label_series) > 0:
                    ic = compute_ic(pred_df, label_series)
                    rank_ic = compute_rank_ic(pred_df, label_series)
                    metrics["ic"] = None if (isinstance(ic, float) and math.isnan(ic)) else ic
                    metrics["rank_ic"] = None if (isinstance(rank_ic, float) and math.isnan(rank_ic)) else rank_ic
            except Exception as exc:
                metrics["ic_error"] = f"{type(exc).__name__}: {exc}"

        # PSI + KS + missing rate need the feature matrix from the same dataset
        baseline_path = Path(args.baseline) if args.baseline else (bundle_dir / "baseline.parquet")
        if not baseline_path.exists():
            metrics["baseline_error"] = f"baseline parquet not found: {baseline_path}"
        elif dataset is None:
            metrics["baseline_error"] = "dataset build failed; see dataset_error"
        else:
            try:
                baseline_df = pd.read_parquet(baseline_path)
                features_df = dataset.prepare(["test"], col_set="feature")[0]
                if isinstance(features_df.columns, pd.MultiIndex):
                    features_df.columns = [c[-1] if isinstance(c, tuple) else c for c in features_df.columns]

                psi_by_feature = compute_psi_by_feature(features_df, baseline_df)
                metrics["psi_by_feature"] = psi_by_feature
                metrics.update(summarize_psi(psi_by_feature))
                metrics["ks_by_feature"] = compute_ks_by_feature(features_df, baseline_df)
                metrics["feat_missing"] = compute_feature_missing_rate(features_df)
            except Exception as exc:
                metrics["baseline_error"] = f"{type(exc).__name__}: {exc}"

        _atomic_write_json(out_dir / "metrics.json", metrics)

        # 3. diagnostics (SENTINEL — last write)
        # rows/status 用当日 pred (与 predictions.parquet/n_predictions 口径一致);
        # pred_df (整段 test segment) 只留做 IC/PSI/KS 计算输入.
        diag_base.update({
            "status": "ok" if len(pred_df_today) > 0 else "empty",
            "exit_code": 0,
            "duration_ms": int((time.time() - t0) * 1000),
            "rows": int(len(pred_df_today)),
            "model_run_id": model_run_id,
            "core_version": core_version,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        })
        _atomic_write_json(diag_path, diag_base)

        print(f"[run_inference] {args.strategy}@{args.live_end} → {len(pred_df_today)} rows (of {len(pred_df)} total pred_df) in {diag_base['duration_ms']}ms")
        return 0

    except Exception as exc:
        # Write diagnostics even on failure so the main process can distinguish
        # "subprocess ran and failed" from "subprocess never started".
        diag_base.update({
            "status": "failed",
            "exit_code": 1,
            "duration_ms": int((time.time() - t0) * 1000),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        })
        try:
            _atomic_write_json(diag_path, diag_base)
        except Exception:
            pass
        print(f"[run_inference] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
