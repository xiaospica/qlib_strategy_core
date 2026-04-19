"""Probe: predict_from_bundle + dataset.prepare('label') 对齐检查.

## 用途

调试 IC 计算. 当 subprocess 跑出 metrics.ic=null 时用它定位原因:
  - pred_df datetime 范围
  - label_series datetime 范围
  - 是否有共同日期
  - 每日 non-nan label 数量

## 运行 (必须起 `__main__` 守卫, 否则 Windows qlib spawn fork bomb)

```
cd /f/Quant/code/qlib_strategy_dev
E:/ssd_backup/Pycharm_project/python-3.11.0-amd64/python.exe -u \
  vendor/qlib_strategy_core/scripts/probe_label_dataset.py
```

## 前置

  - 同 subprocess: bundle + qlib_data_bin + Python 3.11 qlib env
"""
import sys, os
from pathlib import Path

# Script at qlib_strategy_core/scripts/ — insert parent so package resolves
# when running without `pip install -e`.
_CORE_ROOT = Path(__file__).resolve().parents[1]
if str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

os.environ.setdefault(
    "QLIB_PROVIDER_URI",
    r"F:/Quant/code/qlib_strategy_dev/factor_factory/qlib_data_bin",
)

from qlib_strategy_core.pipeline import RollingEnv, TaskBuilder
from qlib_strategy_core.inference import predict_from_bundle
import pandas as pd


def main() -> None:
    RollingEnv.setup_env()
    RollingEnv.init_qlib(provider_uri=r"F:/Quant/code/qlib_strategy_dev/factor_factory/qlib_data_bin")

    live_end = pd.Timestamp("2026-01-10").normalize()
    pred_df, task = predict_from_bundle(
        bundle_dir=r"F:/Quant/code/qlib_strategy_dev/qs_exports/rolling_exp/ab2711178313491f9900b5695b47fa98",
        live_end=live_end,
        lookback_days=60,
    )
    print("pred_df datetimes unique:", pred_df.index.get_level_values("datetime").nunique())
    pred_tmax = pred_df.index.get_level_values("datetime").max()
    print("pred_df tmax:", pred_tmax, "rows at tmax:", len(pred_df.xs(pred_tmax, level="datetime")))

    ds = TaskBuilder.build_dataset_from_task(task)
    lbl = ds.prepare(["test"], col_set="label")[0]
    print("\nlabel type:", type(lbl).__name__, "shape:", getattr(lbl, "shape", "N/A"))
    if hasattr(lbl, "columns"):
        print("label cols:", list(lbl.columns))

    lbl_s = lbl.iloc[:, 0] if hasattr(lbl, "shape") and len(lbl.shape) == 2 else lbl
    print("label total rows:", len(lbl_s), "non-nan:", lbl_s.notna().sum())
    print("label datetime range:", lbl_s.index.get_level_values("datetime").min(), "→", lbl_s.index.get_level_values("datetime").max())

    nn = lbl_s.groupby(lbl_s.index.get_level_values("datetime")).apply(lambda s: s.notna().sum())
    print("\nlabel non-nan per date (last 15):")
    print(nn.tail(15))

    try:
        y_at_tmax = lbl_s.xs(pred_tmax, level="datetime")
        print(f"\nlabel.xs({pred_tmax}): {len(y_at_tmax)} rows, non-nan: {y_at_tmax.notna().sum()}")
    except KeyError:
        print(f"\nlabel.xs({pred_tmax}): KeyError")

    lbl_dropna = lbl_s.dropna()
    try:
        y_at_tmax = lbl_dropna.xs(pred_tmax, level="datetime")
        print(f"label.dropna().xs({pred_tmax}): {len(y_at_tmax)} rows")
    except KeyError:
        print(f"label.dropna().xs({pred_tmax}): KeyError — this is why IC=nan")

    print("\nlabel.dropna() last 10 datetimes:")
    print(lbl_dropna.index.get_level_values("datetime").unique().sort_values()[-10:])


if __name__ == "__main__":
    main()
