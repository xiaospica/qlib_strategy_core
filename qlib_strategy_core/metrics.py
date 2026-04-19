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


def compute_ic(pred_df: pd.DataFrame, label_series: pd.Series) -> float:
    """Latest-day Pearson IC.

    Parameters
    ----------
    pred_df : DataFrame
        MultiIndex (datetime, instrument), column ``score``. Uses only the
        max datetime in the index.
    label_series : Series
        MultiIndex (datetime, instrument), realized returns (T+1 or whatever
        horizon the model targets).
    """
    if pred_df.empty or label_series.empty:
        return float("nan")

    t = pred_df.index.get_level_values("datetime").max()
    try:
        p = pred_df.xs(t, level="datetime")["score"]
        y = label_series.xs(t, level="datetime")
    except KeyError:
        return float("nan")
    return _cross_section_corr(p, y, method="pearson")


def compute_rank_ic(pred_df: pd.DataFrame, label_series: pd.Series) -> float:
    """Latest-day Spearman rank IC. Same shape as :func:`compute_ic`."""
    if pred_df.empty or label_series.empty:
        return float("nan")

    t = pred_df.index.get_level_values("datetime").max()
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
