"""Feature distribution baseline — used for PSI monitoring on the inference side.

At training time, ``compute_feature_distribution`` samples the last N days of
training-period features and produces a long-format Parquet with quantile bins
per feature. The inference side reads this Parquet and computes PSI by scoring
today's feature distribution against the frozen training bins.

Schema (long format, one row per (feature, bin)):

    feature     str         feature column name
    bin_id      int         0..n_bins-1
    edge_lo     float       left edge (inclusive), -inf for bin 0
    edge_hi     float       right edge (exclusive), +inf for last bin
    probability float       fraction of training samples in this bin

The ``BASELINE_SCHEMA_VERSION`` lets both sides evolve the format safely.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


BASELINE_SCHEMA_VERSION = 1

# Column names — consumers should import these constants instead of hardcoding.
COL_FEATURE = "feature"
COL_BIN_ID = "bin_id"
COL_EDGE_LO = "edge_lo"
COL_EDGE_HI = "edge_hi"
COL_PROBABILITY = "probability"


def compute_feature_distribution(
    features_df: pd.DataFrame,
    n_bins: int = 20,
    feature_names: Optional[Iterable[str]] = None,
    min_samples: int = 100,
) -> pd.DataFrame:
    """Compute per-feature quantile bins + probabilities.

    Parameters
    ----------
    features_df : DataFrame
        Wide feature matrix. Columns are feature names; rows are samples
        (typically MultiIndex (datetime, instrument) but any row index works).
    n_bins : int
        Number of quantile bins. Default 20 (matches PSI industry standard).
    feature_names : iterable[str], optional
        Subset of columns to process. Defaults to all columns.
    min_samples : int
        Features with fewer than this many non-NaN samples are skipped.

    Returns
    -------
    DataFrame with columns ``[feature, bin_id, edge_lo, edge_hi, probability]``.
    Rows: n_features × n_bins (approximately — features with few unique values
    may produce fewer effective bins).
    """
    if feature_names is None:
        feature_names = features_df.columns

    rows = []
    for feat in feature_names:
        series = features_df[feat].dropna()
        if len(series) < min_samples:
            continue

        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = series.quantile(quantiles).to_numpy()
        edges = np.unique(edges)
        if len(edges) < 2:
            continue

        edges[0] = -np.inf
        edges[-1] = np.inf

        bin_assignments = pd.cut(series, bins=edges, include_lowest=True, labels=False)
        counts = bin_assignments.value_counts(sort=False).sort_index()
        probs = counts / counts.sum()

        for bin_id in range(len(edges) - 1):
            prob = float(probs.get(bin_id, 0.0))
            rows.append(
                {
                    COL_FEATURE: str(feat),
                    COL_BIN_ID: int(bin_id),
                    COL_EDGE_LO: float(edges[bin_id]),
                    COL_EDGE_HI: float(edges[bin_id + 1]),
                    COL_PROBABILITY: prob,
                }
            )

    return pd.DataFrame(rows, columns=[COL_FEATURE, COL_BIN_ID, COL_EDGE_LO, COL_EDGE_HI, COL_PROBABILITY])
