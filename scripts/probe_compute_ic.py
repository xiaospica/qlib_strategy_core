"""Probe: 单步验证 compute_ic / compute_rank_ic 是否返回非 nan.

## 用途

当 probe_label_dataset 确认 pred 和 label 都存在共同日期后, 用这个验证
core.metrics 里的 _latest_common_date / compute_ic 真正走通:
  - 打印 _latest_common_date 返回的 t
  - 打印 pred.xs(t) 和 label.xs(t) 的 shape/nunique
  - 打印 concat.dropna 后 p/y 的 nunique (必须 >=2 否则 corr 返 nan)
  - 直接调 compute_ic / compute_rank_ic 打印返回值

## 运行

```
cd /f/Quant/code/qlib_strategy_dev
E:/ssd_backup/Pycharm_project/python-3.11.0-amd64/python.exe -u \
  vendor/qlib_strategy_core/scripts/probe_compute_ic.py
```
"""
import sys, os
from pathlib import Path

_CORE_ROOT = Path(__file__).resolve().parents[1]
if str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

os.environ.setdefault(
    "QLIB_PROVIDER_URI",
    r"F:/Quant/code/qlib_strategy_dev/factor_factory/qlib_data_bin",
)

from qlib_strategy_core.pipeline import RollingEnv, TaskBuilder
from qlib_strategy_core.inference import predict_from_bundle
from qlib_strategy_core.metrics import _latest_common_date, compute_ic, compute_rank_ic, _cross_section_corr
import pandas as pd


def main() -> None:
    RollingEnv.setup_env()
    RollingEnv.init_qlib(provider_uri=r"F:/Quant/code/qlib_strategy_dev/factor_factory/qlib_data_bin")

    pred_df, task = predict_from_bundle(
        bundle_dir=r"F:/Quant/code/qlib_strategy_dev/qs_exports/rolling_exp/ab2711178313491f9900b5695b47fa98",
        live_end=pd.Timestamp("2026-01-10").normalize(),
        lookback_days=60,
    )
    print("pred_df shape:", pred_df.shape)
    print("pred_df index names:", pred_df.index.names)
    print("pred_df datetime max:", pred_df.index.get_level_values("datetime").max())

    ds = TaskBuilder.build_dataset_from_task(task)
    lbl_raw = ds.prepare(["test"], col_set="label")[0]
    label_series = lbl_raw.iloc[:, 0] if hasattr(lbl_raw, "shape") and len(lbl_raw.shape) == 2 else lbl_raw
    label_series = label_series.dropna()
    print("\nlabel_series shape:", label_series.shape)
    print("label_series index names:", label_series.index.names)
    print("label_series datetime max:", label_series.index.get_level_values("datetime").max())
    print("label_series name:", label_series.name)

    # latest common
    t = _latest_common_date(pred_df, label_series)
    print(f"\n_latest_common_date -> {t!r} (type={type(t).__name__})")

    if t is not None:
        p = pred_df.xs(t, level="datetime")["score"]
        y = label_series.xs(t, level="datetime")
        print(f"p shape={p.shape} nunique={p.nunique()}")
        print(f"y shape={y.shape} nunique={y.nunique()}")
        print(f"p head:\n{p.head(3)}")
        print(f"y head:\n{y.head(3)}")
        df = pd.concat([p.rename("p"), y.rename("y")], axis=1).dropna()
        print(f"after concat+dropna: {df.shape}, p nunique={df['p'].nunique()}, y nunique={df['y'].nunique()}")
        corr_pearson = df["p"].corr(df["y"], method="pearson")
        corr_spearman = df["p"].corr(df["y"], method="spearman")
        print(f"pearson corr: {corr_pearson}")
        print(f"spearman corr: {corr_spearman}")

    print(f"\ncompute_ic(pred_df, label_series) = {compute_ic(pred_df, label_series)}")
    print(f"compute_rank_ic(pred_df, label_series) = {compute_rank_ic(pred_df, label_series)}")


if __name__ == "__main__":
    main()
