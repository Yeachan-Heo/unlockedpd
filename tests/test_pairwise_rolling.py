"""Focused tests for rolling pairwise corr/cov resource guards."""
import numpy as np
import pandas as pd
import pytest

import unlockedpd
from unlockedpd.ops import pairwise as pairwise_ops


@pytest.fixture(autouse=True)
def reset_unlockedpd_config():
    original_enabled = unlockedpd.config.enabled
    original_warn_on_fallback = unlockedpd.config.warn_on_fallback
    yield
    unlockedpd.config.enabled = original_enabled
    unlockedpd.config.warn_on_fallback = original_warn_on_fallback


def _pandas_rolling_pairwise(df, op_name, window=3):
    unlockedpd.config.enabled = False
    try:
        return getattr(df.rolling(window), op_name)()
    finally:
        unlockedpd.config.enabled = True


@pytest.mark.parametrize("op_name", ["corr", "cov"])
def test_rolling_pairwise_matches_pandas_and_preserves_labels(op_name):
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        rng.normal(size=(8, 4)),
        index=pd.Index(range(10, 18), name="row"),
        columns=pd.Index(list("abcd"), name="asset"),
    )
    expected = _pandas_rolling_pairwise(df, op_name)

    unlockedpd.config.enabled = True
    result = getattr(df.rolling(3), op_name)()

    pd.testing.assert_frame_equal(result, expected, rtol=1e-10, atol=1e-12)


def test_rolling_corr_constant_diagonal_matches_pandas_nan_semantics():
    df = pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0, 1.0],
            "varying": [1.0, 2.0, 3.0, 4.0],
        }
    )
    expected = _pandas_rolling_pairwise(df, "corr")

    unlockedpd.config.enabled = True
    result = df.rolling(3).corr()

    pd.testing.assert_frame_equal(result, expected, rtol=1e-10, atol=1e-12)
    assert np.isnan(result.loc[(2, "constant"), "constant"])


def test_memory_guard_falls_back_to_pandas_before_pairwise_allocation(monkeypatch):
    df = pd.DataFrame(np.arange(24, dtype=np.float64).reshape(6, 4))
    expected = _pandas_rolling_pairwise(df, "cov")
    monkeypatch.setattr(pairwise_ops, "_available_memory_bytes", lambda: 1)
    unlockedpd.config.warn_on_fallback = True
    unlockedpd.config.enabled = True

    with pytest.warns(RuntimeWarning, match="pairwise rolling memory guard"):
        result = df.rolling(3).cov()

    pd.testing.assert_frame_equal(result, expected, rtol=1e-10, atol=1e-12)


def test_pairwise_memory_estimate_avoids_redundant_output_sized_scratch():
    estimate = pairwise_ops._estimate_pairwise_memory(
        n_rows=100,
        n_cols=10,
        input_nbytes=100 * 10 * 8,
        input_copy_bytes=0,
    )

    assert estimate["output_bytes"] == 100 * 10 * 10 * 8
    assert estimate["scratch_bytes"] < estimate["output_bytes"] // 10
