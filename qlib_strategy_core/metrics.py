"""ML monitoring metrics — shared between training-side verification and
vnpy-side inference subprocess.

All functions are pure: (arrays/DataFrames) → (float/dict/list). No I/O, no
logging, no side effects. Callers handle persistence.

Conventions
-----------
* Prediction DataFrames carry MultiIndex ``(datetime, instrument)`` with a
  ``score`` column. Feature DataFrames are wide — one column per feature,
  same MultiIndex.
* PSI compares a *live* feature distribution against a *baseline* quantile
  distribution produced by :func:`qlib_strategy_core.baseline.compute_feature_distribution`.
* All outputs are JSON-safe (float, int, str, list, dict of same) — the
  inference subprocess serializes these directly into ``metrics.json``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from qlib_strategy_core.baseline import (
    COL_FEATURE,
    COL_BIN_ID,
    COL_EDGE_LO,
    COL_EDGE_HI,
    COL_PROBABILITY,
)


# ---------------------------------------------------------------------------
# Cross-section IC (Pearson) + RankIC (Spearman)
# ---------------------------------------------------------------------------


def _cross_section_corr(
    pred: pd.Series,
    label: pd.Series,
    method: str,
) -> float:
    """Compute cross-section correlation for a single trade date.

    Both series must be aligned on the ``instrument`` index. NaNs are dropped
    pairwise. Returns NaN if fewer than 5 instruments remain.
    """
    df = pd.concat({"p": pred, "y": label}, axis=1).dropna()
    if len(df) < 5:
        return float("nan")
    if df["p"].nunique() < 2 or df["y"].nunique() < 2:
        return float("nan")
    return float(df["p"].corr(df["y"], method=method))


def _latest_common_date(pred_df: pd.DataFrame, label_series: pd.Series):
    """Most recent date present in both pred_df and label_series.

    label_series often truncates N days earlier than pred_df because labels
    like ``Ref($close, -11) / Ref($close, -1) - 1`` can't evaluate for the
    last N trading days (forward window exceeds available data). Using
    ``pred_df.datetime.max()`` directly would KeyError on label. Instead
    fall back to the latest date where both sides have rows.
    """
    pred_dates = set(pred_df.index.get_level_values("datetime").unique())
    label_dates = set(label_series.index.get_level_values("datetime").unique())
    common = pred_dates & label_dates
    if not common:
        return None
    return max(common)


def compute_ic(pred_df: pd.DataFrame, label_series: pd.Series) -> float:
    """Latest-common-day Pearson IC.

    Uses the most recent date present in BOTH pred_df and label_series. If
    label_series is missing the last N days (due to forward-return horizon),
    the IC is computed on the latest day that has both prediction and label.

    Parameters
    ----------
    pred_df : DataFrame
        MultiIndex (datetime, instrument), column ``score``.
    label_series : Series
        MultiIndex (datetime, instrument), realized returns (T+1 or whatever
        horizon the model targets).
    """
    if pred_df.empty or label_series.empty:
        return float("nan")
    t = _latest_common_date(pred_df, label_series)
    if t is None:
        return float("nan")
    try:
        p = pred_df.xs(t, level="datetime")["score"]
        y = label_series.xs(t, level="datetime")
    except KeyError:
        return float("nan")
    return _cross_section_corr(p, y, method="pearson")


def compute_rank_ic(pred_df: pd.DataFrame, label_series: pd.Series) -> float:
    """Latest-common-day Spearman rank IC. Same shape as :func:`compute_ic`."""
    if pred_df.empty or label_series.empty:
        return float("nan")
    t = _latest_common_date(pred_df, label_series)
    if t is None:
        return float("nan")
    try:
        p = pred_df.xs(t, level="datetime")["score"]
        y = label_series.xs(t, level="datetime")
    except KeyError:
        return float("nan")
    return _cross_section_corr(p, y, method="spearman")


# ---------------------------------------------------------------------------
# Population Stability Index (PSI) — per-feature vs. training-period baseline
# ---------------------------------------------------------------------------


def _psi_single(live: np.ndarray, baseline_edges: np.ndarray, baseline_probs: np.ndarray) -> float:
    """PSI for one feature given live values + baseline bins.

    Formula: sum_i (p_live_i - p_base_i) * ln(p_live_i / p_base_i)
    with smoothing epsilon 1e-6 to avoid log(0).
    """
    live = live[~np.isnan(live)]
    if len(live) == 0:
        return float("nan")

    bin_counts = np.zeros(len(baseline_probs))
    bin_assign = np.searchsorted(baseline_edges[1:-1], live, side="right")
    for b in bin_assign:
        if 0 <= b < len(bin_counts):
            bin_counts[b] += 1

    live_probs = bin_counts / bin_counts.sum() if bin_counts.sum() > 0 else bin_counts
    eps = 1e-6
    p_live = np.clip(live_probs, eps, 1.0)
    p_base = np.clip(baseline_probs, eps, 1.0)
    return float(np.sum((p_live - p_base) * np.log(p_live / p_base)))


def compute_psi_by_feature(
    features_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    feature_names: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Compute PSI for each feature in ``features_df`` vs. baseline bins.

    Parameters
    ----------
    features_df : DataFrame
        Wide live feature matrix (any index). Column names are features.
    baseline_df : DataFrame
        Long-form baseline with columns from :mod:`qlib_strategy_core.baseline`
        (``feature``, ``bin_id``, ``edge_lo``, ``edge_hi``, ``probability``).
    feature_names : iterable[str], optional
        Subset to compute; defaults to intersection of live cols and baseline features.

    Returns
    -------
    ``{feature_name: psi_value}``. Features missing from baseline are skipped.
    """
    result: Dict[str, float] = {}
    baseline_features = set(baseline_df[COL_FEATURE].unique())

    if feature_names is None:
        feature_names = [c for c in features_df.columns if c in baseline_features]

    for feat in feature_names:
        if feat not in baseline_features or feat not in features_df.columns:
            continue

        sub = baseline_df[baseline_df[COL_FEATURE] == feat].sort_values(COL_BIN_ID)
        edges = np.concatenate([[sub[COL_EDGE_LO].iloc[0]], sub[COL_EDGE_HI].to_numpy()])
        probs = sub[COL_PROBABILITY].to_numpy()

        live_vals = features_df[feat].to_numpy(dtype=float)
        result[feat] = _psi_single(live_vals, edges, probs)

    return result


def summarize_psi(psi_by_feature: Dict[str, float]) -> Dict[str, float]:
    """Produce {psi_mean, psi_max, psi_count, psi_n_over_0_25}."""
    if not psi_by_feature:
        return {"psi_mean": float("nan"), "psi_max": float("nan"), "psi_count": 0, "psi_n_over_0_25": 0}
    values = np.array([v for v in psi_by_feature.values() if not np.isnan(v)])
    if len(values) == 0:
        return {"psi_mean": float("nan"), "psi_max": float("nan"), "psi_count": 0, "psi_n_over_0_25": 0}
    return {
        "psi_mean": float(np.mean(values)),
        "psi_max": float(np.max(values)),
        "psi_count": int(len(values)),
        "psi_n_over_0_25": int(np.sum(values > 0.25)),
    }


# ---------------------------------------------------------------------------
# Kolmogorov–Smirnov feature-distribution distance
# ---------------------------------------------------------------------------


def compute_ks_by_feature(
    features_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    feature_names: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Per-feature KS statistic approximated from the baseline's empirical CDF.

    We reconstruct the baseline CDF from the bin edges + probabilities, then
    compute sup |F_live(x) - F_base(x)| over a shared grid.
    """
    result: Dict[str, float] = {}
    baseline_features = set(baseline_df[COL_FEATURE].unique())
    if feature_names is None:
        feature_names = [c for c in features_df.columns if c in baseline_features]

    for feat in feature_names:
        if feat not in baseline_features or feat not in features_df.columns:
            continue
        sub = baseline_df[baseline_df[COL_FEATURE] == feat].sort_values(COL_BIN_ID)
        edges = np.concatenate([[sub[COL_EDGE_LO].iloc[0]], sub[COL_EDGE_HI].to_numpy()])
        cum_probs = np.cumsum(sub[COL_PROBABILITY].to_numpy())

        live_vals = features_df[feat].dropna().to_numpy(dtype=float)
        if len(live_vals) < 5:
            continue
        live_sorted = np.sort(live_vals)

        # evaluate both CDFs on bin edges (excluding infinities)
        finite_edges = edges[np.isfinite(edges)]
        if len(finite_edges) < 2:
            continue
        live_cdf = np.searchsorted(live_sorted, finite_edges, side="right") / len(live_sorted)
        base_cdf = np.interp(
            finite_edges,
            edges[:-1][np.isfinite(edges[:-1])] if np.all(np.isfinite(edges[:-1])) else edges[1:],
            cum_probs[:len(np.isfinite(edges[:-1]))] if False else cum_probs,
            left=0.0,
            right=1.0,
        )
        if len(live_cdf) == len(base_cdf):
            result[feat] = float(np.max(np.abs(live_cdf - base_cdf)))
    return result


# ---------------------------------------------------------------------------
# Prediction stats + histogram + feature missing rate
# ---------------------------------------------------------------------------


def compute_prediction_stats(pred_df: pd.DataFrame) -> Dict[str, float]:
    """mean/std/quantiles/zero-ratio of the latest-day prediction scores."""
    if pred_df.empty:
        return {}
    t = pred_df.index.get_level_values("datetime").max()
    scores = pred_df.xs(t, level="datetime")["score"].to_numpy()
    if len(scores) == 0:
        return {}
    return {
        "pred_mean": float(np.mean(scores)),
        "pred_std": float(np.std(scores)),
        "pred_q05": float(np.quantile(scores, 0.05)),
        "pred_q50": float(np.quantile(scores, 0.5)),
        "pred_q95": float(np.quantile(scores, 0.95)),
        "pred_min": float(np.min(scores)),
        "pred_max": float(np.max(scores)),
        "pred_zero_ratio": float(np.sum(np.abs(scores) < 1e-9) / len(scores)),
    }


def compute_score_histogram(pred_df: pd.DataFrame, n_bins: int = 20) -> List[Dict[str, float]]:
    """Latest-day score histogram, ``n_bins`` equal-width buckets.

    Output: ``[{bin_id, edge_lo, edge_hi, count, probability}, ...]``.
    """
    if pred_df.empty:
        return []
    t = pred_df.index.get_level_values("datetime").max()
    scores = pred_df.xs(t, level="datetime")["score"].to_numpy()
    if len(scores) == 0:
        return []
    counts, edges = np.histogram(scores, bins=n_bins)
    probs = counts / counts.sum() if counts.sum() > 0 else counts
    return [
        {
            "bin_id": int(i),
            "edge_lo": float(edges[i]),
            "edge_hi": float(edges[i + 1]),
            "count": int(counts[i]),
            "probability": float(probs[i]),
        }
        for i in range(len(counts))
    ]


def compute_feature_missing_rate(features_df: pd.DataFrame) -> Dict[str, float]:
    """Per-feature NaN ratio — cheap sanity check for upstream data issues."""
    if features_df.empty:
        return {}
    n = len(features_df)
    return {
        str(col): float(features_df[col].isna().sum() / n)
        for col in features_df.columns
    }


# ---------------------------------------------------------------------------
# Cross-day / time-series aggregation (called by monitoring server,
# not by subprocess inference). Single source of truth for IC / PSI trend
# analysis — downstream (mlearnweb ml_aggregation_service) imports these.
# ---------------------------------------------------------------------------


def compute_icir(
    metrics_series: List[Dict[str, object]],
    window: int,
) -> Dict[str, Optional[float]]:
    """Rolling ICIR = mean(ic) / std(ic) over the last ``window`` snapshots.

    metrics_series: list of snapshots each with at least an "ic" key (None
    allowed). Ordering: oldest first → most recent last (same as
    ``ml_metric_snapshots`` ORDER BY trade_date ASC).

    Returns ``{window, icir, ic_mean, ic_std, n_samples}``.
    """
    recent = metrics_series[-window:] if len(metrics_series) > window else metrics_series
    ics = [m.get("ic") for m in recent]
    valid = [v for v in ics if v is not None]
    if len(valid) < 2:
        mean = valid[0] if valid else None
        return {"window": window, "icir": None, "ic_mean": mean, "ic_std": None, "n_samples": len(valid)}
    mean = sum(valid) / len(valid)
    var = sum((v - mean) ** 2 for v in valid) / (len(valid) - 1)
    std = var ** 0.5
    icir = (mean / std) if std > 0 else None
    return {"window": window, "icir": icir, "ic_mean": mean, "ic_std": std, "n_samples": len(valid)}


def psi_trend_alerts(
    metrics_series: List[Dict[str, object]],
    threshold: float = 0.25,
    consecutive_days: int = 3,
) -> Dict[str, object]:
    """连续 N 日 psi_mean > threshold 触发告警.

    Returns ``{triggered, threshold, consecutive_days, last_streak_days,
               max_streak_days, first_alert_date}``.
    """
    streak = 0
    max_streak = 0
    first_alert_date: Optional[str] = None
    for m in metrics_series:
        psi = m.get("psi_mean")
        if psi is not None and psi > threshold:
            streak += 1
            if streak >= consecutive_days and first_alert_date is None:
                first_alert_date = str(m.get("trade_date", ""))[:10]
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return {
        "triggered": max_streak >= consecutive_days,
        "threshold": threshold,
        "consecutive_days": consecutive_days,
        "last_streak_days": streak,
        "max_streak_days": max_streak,
        "first_alert_date": first_alert_date,
    }


def detect_ic_decay(
    metrics_series: List[Dict[str, object]],
    min_samples: int = 10,
    decay_ratio_threshold: float = 0.5,
    absolute_floor: float = 0.02,
) -> Dict[str, object]:
    """Detect material degradation in daily IC vs an earlier window.

    Heuristic (no t-test, small-sample friendly):
      1. Split samples by mid → prior half / recent half
      2. Trigger A (衰减): prior_mean > 0 AND recent_mean < prior_mean * decay_ratio_threshold
      3. Trigger B (崩盘): prior_mean >= absolute_floor AND recent_mean < absolute_floor

    Returns ``{triggered, reason, recent_ic_mean, prior_ic_mean, decay_ratio, n_recent, n_prior}``.
    """
    vals = [m.get("ic") for m in metrics_series]
    valid = [v for v in vals if v is not None]
    if len(valid) < min_samples:
        return {
            "triggered": False,
            "reason": f"样本不足 ({len(valid)}/{min_samples})",
            "recent_ic_mean": None, "prior_ic_mean": None, "decay_ratio": None,
            "n_recent": 0, "n_prior": 0,
        }
    mid = len(valid) // 2
    prior = valid[:mid]
    recent = valid[mid:]
    prior_mean = sum(prior) / len(prior)
    recent_mean = sum(recent) / len(recent)
    decay_ratio = (recent_mean / prior_mean) if prior_mean not in (0, None) else None

    if prior_mean >= absolute_floor and recent_mean < absolute_floor:
        return {
            "triggered": True,
            "reason": f"IC 均值跌破 {absolute_floor} (近期 {recent_mean:.4f} vs 前期 {prior_mean:.4f})",
            "recent_ic_mean": recent_mean, "prior_ic_mean": prior_mean,
            "decay_ratio": decay_ratio,
            "n_recent": len(recent), "n_prior": len(prior),
        }
    if prior_mean > 0 and recent_mean < prior_mean * decay_ratio_threshold:
        return {
            "triggered": True,
            "reason": f"IC 较前期下滑 {(1 - decay_ratio) * 100:.1f}% (近期 {recent_mean:.4f} vs 前期 {prior_mean:.4f})",
            "recent_ic_mean": recent_mean, "prior_ic_mean": prior_mean,
            "decay_ratio": decay_ratio,
            "n_recent": len(recent), "n_prior": len(prior),
        }
    return {
        "triggered": False,
        "reason": "OK",
        "recent_ic_mean": recent_mean, "prior_ic_mean": prior_mean,
        "decay_ratio": decay_ratio,
        "n_recent": len(recent), "n_prior": len(prior),
    }


def compute_live_vs_backtest_diff(
    live_pred: pd.DataFrame,
    backtest_pred: pd.DataFrame,
) -> Dict[str, object]:
    """Compare live prediction DataFrame with training-time backtest prediction.

    Both must be MultiIndex (datetime, instrument) with a ``score`` column.

    Returns per-date correlation + aggregate coverage + cumulative gap metrics:
      {
        "per_date": [{"trade_date", "corr", "mean_abs_diff", "coverage", "n_overlap"}],
        "coverage_ratio": <live-dates with backtest data / live-dates total>,
        "corr_mean": <pearson of per-date corr>,
        "n_dates_in_overlap": int,
      }
    """
    if live_pred.empty or backtest_pred.empty:
        return {
            "per_date": [], "coverage_ratio": 0.0,
            "corr_mean": None, "n_dates_in_overlap": 0,
        }

    live_dates = sorted(set(live_pred.index.get_level_values("datetime").unique()))
    bt_dates = set(backtest_pred.index.get_level_values("datetime").unique())
    overlap_dates = [d for d in live_dates if d in bt_dates]

    per_date: List[Dict[str, object]] = []
    for t in overlap_dates:
        try:
            lp = live_pred.xs(t, level="datetime")["score"]
            bp = backtest_pred.xs(t, level="datetime")["score"]
        except KeyError:
            continue
        joined = pd.concat({"l": lp, "b": bp}, axis=1).dropna()
        if len(joined) < 5:
            continue
        try:
            corr = float(joined["l"].corr(joined["b"], method="pearson"))
        except Exception:
            corr = None
        mean_abs_diff = float((joined["l"] - joined["b"]).abs().mean())
        per_date.append({
            "trade_date": str(pd.Timestamp(t).date()),
            "corr": corr,
            "mean_abs_diff": mean_abs_diff,
            "coverage": float(len(joined) / max(len(lp), 1)),
            "n_overlap": int(len(joined)),
        })

    corrs = [p["corr"] for p in per_date if p["corr"] is not None]
    corr_mean = (sum(corrs) / len(corrs)) if corrs else None
    coverage_ratio = (len(overlap_dates) / max(len(live_dates), 1)) if live_dates else 0.0

    return {
        "per_date": per_date,
        "coverage_ratio": coverage_ratio,
        "corr_mean": corr_mean,
        "n_dates_in_overlap": len(per_date),
    }
