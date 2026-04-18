"""Unit tests for compute_feature_distribution."""

import numpy as np
import pandas as pd

from qlib_strategy_core.baseline import (
    compute_feature_distribution,
    COL_FEATURE,
    COL_BIN_ID,
    COL_EDGE_LO,
    COL_EDGE_HI,
    COL_PROBABILITY,
)


def test_basic_distribution_uniform():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "x": rng.uniform(0, 1, size=10_000),
        "y": rng.normal(0, 1, size=10_000),
    })

    result = compute_feature_distribution(df, n_bins=10)

    expected_cols = {COL_FEATURE, COL_BIN_ID, COL_EDGE_LO, COL_EDGE_HI, COL_PROBABILITY}
    assert expected_cols.issubset(set(result.columns))

    # Two features, ~10 bins each
    assert set(result[COL_FEATURE].unique()) == {"x", "y"}
    for feat in ("x", "y"):
        probs = result.loc[result[COL_FEATURE] == feat, COL_PROBABILITY]
        assert abs(probs.sum() - 1.0) < 1e-9
        # Uniform quantiles → each bin ~0.1
        assert all(abs(p - 0.1) < 0.02 for p in probs)


def test_skip_low_sample_features():
    sparse_vals = np.full(1000, np.nan)
    sparse_vals[:5] = [1, 2, 3, 4, 5]
    df = pd.DataFrame({
        "sparse": sparse_vals,
        "dense": np.arange(1000, dtype=float),
    })

    result = compute_feature_distribution(df, n_bins=10, min_samples=100)
    assert "sparse" not in result[COL_FEATURE].values
    assert "dense" in result[COL_FEATURE].values


def test_edges_cover_minus_plus_inf():
    df = pd.DataFrame({"x": np.arange(1000)})
    result = compute_feature_distribution(df, n_bins=5)

    first_edge_lo = result.loc[(result[COL_FEATURE] == "x") & (result[COL_BIN_ID] == 0), COL_EDGE_LO].iloc[0]
    last_bin_id = result.loc[result[COL_FEATURE] == "x", COL_BIN_ID].max()
    last_edge_hi = result.loc[(result[COL_FEATURE] == "x") & (result[COL_BIN_ID] == last_bin_id), COL_EDGE_HI].iloc[0]

    assert first_edge_lo == -np.inf
    assert last_edge_hi == np.inf
