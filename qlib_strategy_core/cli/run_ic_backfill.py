"""IC backfill subprocess entry point — invoked by vnpy main process.

## 背景 (方案 §2.4.5)

实盘 inference 在 T 日产出 metrics.json 时, ic / rank_ic 字段常为 null —
qlib bin 末尾 = T, label = forward_N return 需要 T+1..T+N 日 close, bin 没有.

本 CLI 在后续日期(bin 已补齐到 T+N 之后)把历史 metrics.json 的 ic / rank_ic
回填. 只动这两个字段, 其他字段保持 inference 当日产出.

## 调用

    python -m qlib_strategy_core.cli.run_ic_backfill \
        --output-root  D:/ml_output/smoke_full_pipeline \
        --strategy     phase27_test \
        --provider-uri D:/vnpy_data/qlib_data_bin \
        --scan-days    30 \
        --forward-window 2

## 输出

stdout JSON 一行: ``{"scanned": N, "computed": M, "skipped_no_forward": K, "errors": E, "details": [...]}``
exit 0 = 成功 (即便 computed=0 也算成功); exit 1 = 致命异常.

## 隔离

本进程 import qlib, 主 vnpy 进程不允许 import qlib (方案 §2.2). 调用方必须
通过 subprocess 隔离, 用 inference_python (qlib 兼容的 Python).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="qlib_strategy_core IC backfill — 把历史 metrics.json 里 ic=null 的字段补算"
    )
    p.add_argument("--output-root", required=True, help="{output_root}/{strategy}/{yyyymmdd}/ 根")
    p.add_argument("--strategy", required=True, help="策略名 (扫 {output_root}/{strategy}/ 子目录)")
    p.add_argument("--provider-uri", required=True, help="qlib bin 路径, 用于算 forward return label")
    p.add_argument("--scan-days", type=int, default=30, help="回看自然日数 (扫近 N 天)")
    p.add_argument(
        "--forward-window",
        type=int,
        default=2,
        help="label 的 forward 窗口交易日数 (Alpha158 默认 2 = Ref(close,-2)/Ref(close,-1)-1)",
    )
    p.add_argument(
        "--install-legacy-path",
        action="store_true",
        help="兼容旧 MLflow artifact 模块路径 (factor_factory.*)",
    )
    return p


def _list_metric_files(output_root: Path, strategy: str, scan_days: int) -> List[Path]:
    """枚举近 scan_days 天里存在的 metrics.json. 按日期升序返回."""
    strat_dir = output_root / strategy
    if not strat_dir.exists():
        return []
    cutoff = datetime.now().date()
    files: List[Path] = []
    for sub in sorted(strat_dir.iterdir()):
        if not sub.is_dir():
            continue
        try:
            d = datetime.strptime(sub.name, "%Y%m%d").date()
        except ValueError:
            continue
        if (cutoff - d).days > scan_days:
            continue
        m = sub / "metrics.json"
        if m.exists():
            files.append(m)
    return files


def _is_ic_null(metrics: Dict[str, Any]) -> bool:
    """metrics.json 里 ic / rank_ic 至少一个是 None / 缺失."""
    return metrics.get("ic") is None or metrics.get("rank_ic") is None


def _load_pred_df(day_dir: Path) -> Optional[pd.DataFrame]:
    """读当日 predictions.parquet (期望 MultiIndex (datetime, instrument), 列 'score')."""
    p = day_dir / "predictions.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "score" not in df.columns:
        return None
    return df


def _compute_label_series(
    instruments: List[str],
    trade_date: pd.Timestamp,
    forward_window: int,
) -> Optional[pd.Series]:
    """从 qlib bin 拿 forward return label.

    label 公式与 Alpha158 默认一致: ``Ref($close, -fw) / Ref($close, -1) - 1``,
    在 trade_date 这一天求值, 等价于 ``close[T+fw-1] / close[T+0] - 1``
    (qlib 的 Ref(-N) 是相对当前日期向未来 N 步).

    Returns
    -------
    Series with MultiIndex (datetime, instrument), 单列 = forward return; 若
    qlib bin 末尾日期 < trade_date + forward_window, 返回的 series 在 trade_date
    那行就会全 NaN, 调用方判断后跳过.
    """
    from qlib.data import D  # type: ignore[import-not-found]

    field = f"Ref($close, -{forward_window}) / Ref($close, -1) - 1"
    df = D.features(
        instruments=instruments,
        fields=[field],
        start_time=trade_date,
        end_time=trade_date,
    )
    if df is None or df.empty:
        return None
    # qlib 返回 MultiIndex (instrument, datetime), 列名是 field 字符串.
    # 我们要 (datetime, instrument), 与 pred_df 一致.
    df = df.swaplevel(0, 1).sort_index()
    s = df[field]
    # 若整列全 NaN, 说明 forward window 还没满足 → 调用方跳过本日
    if s.dropna().empty:
        return None
    s.name = "label"
    return s


def _backfill_one(
    metrics_path: Path,
    provider_uri: str,
    forward_window: int,
    qlib_inited: List[bool],
) -> Dict[str, Any]:
    """对单个 metrics.json 尝试回填 ic/rank_ic.

    Returns dict: ``{"trade_date", "status", "ic", "rank_ic", "msg"}``.
    status ∈ {"computed", "skipped_no_forward", "skipped_no_pred",
              "skipped_already_filled", "error"}.
    """
    day_dir = metrics_path.parent
    day_str = day_dir.name  # YYYYMMDD
    rec: Dict[str, Any] = {"trade_date": day_str}
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not _is_ic_null(metrics):
            rec["status"] = "skipped_already_filled"
            return rec

        pred_df = _load_pred_df(day_dir)
        if pred_df is None or pred_df.empty:
            rec["status"] = "skipped_no_pred"
            return rec

        # 当日 instruments 列表
        instruments = pred_df.index.get_level_values("instrument").unique().tolist()
        trade_date = pd.Timestamp(day_str)

        # 懒初始化 qlib (一次进程只 init 一次)
        if not qlib_inited[0]:
            import qlib  # type: ignore[import-not-found]
            qlib.init(provider_uri=provider_uri, region="cn")
            qlib_inited[0] = True

        label_series = _compute_label_series(instruments, trade_date, forward_window)
        if label_series is None:
            rec["status"] = "skipped_no_forward"
            return rec

        from qlib_strategy_core.metrics import compute_ic, compute_rank_ic
        import math
        ic = compute_ic(pred_df, label_series)
        rank_ic = compute_rank_ic(pred_df, label_series)
        ic_v = None if (isinstance(ic, float) and math.isnan(ic)) else float(ic)
        rank_ic_v = None if (isinstance(rank_ic, float) and math.isnan(rank_ic)) else float(rank_ic)

        if ic_v is None and rank_ic_v is None:
            # 都没算出来 (比如 pred 全 NaN), 不动文件
            rec["status"] = "skipped_no_forward"
            return rec

        # 原子重写 metrics.json (只动 ic / rank_ic, 其他字段保持)
        metrics["ic"] = ic_v
        metrics["rank_ic"] = rank_ic_v
        metrics["ic_backfilled_at"] = datetime.now().isoformat(timespec="seconds")
        _atomic_write_json(metrics_path, metrics)

        rec.update({"status": "computed", "ic": ic_v, "rank_ic": rank_ic_v})
        return rec
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "error"
        rec["msg"] = f"{type(exc).__name__}: {exc}"
        return rec


def main() -> int:
    args = _build_arg_parser().parse_args()
    output_root = Path(args.output_root)

    if args.install_legacy_path:
        from qlib_strategy_core._compat import install_finder
        install_finder()

    files = _list_metric_files(output_root, args.strategy, args.scan_days)

    summary: Dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "scanned": len(files),
        "computed": 0,
        "skipped_no_forward": 0,
        "skipped_no_pred": 0,
        "skipped_already_filled": 0,
        "errors": 0,
        "details": [],
    }
    qlib_inited = [False]
    t0 = time.time()
    try:
        for mp in files:
            rec = _backfill_one(mp, args.provider_uri, args.forward_window, qlib_inited)
            summary["details"].append(rec)
            status = rec["status"]
            key = status if status in (
                "computed", "skipped_no_forward", "skipped_no_pred", "skipped_already_filled", "errors",
            ) else None
            if status == "error":
                summary["errors"] += 1
            elif key is not None:
                summary[key] += 1
        summary["duration_ms"] = int((time.time() - t0) * 1000)
        print(json.dumps(summary, ensure_ascii=False, default=str))
        return 0
    except Exception as exc:  # noqa: BLE001
        summary["fatal"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        summary["duration_ms"] = int((time.time() - t0) * 1000)
        print(json.dumps(summary, ensure_ascii=False, default=str), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
