"""Unit tests for the Diebold–Mariano test and the stationary bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.stats import (
    bootstrap_sharpe_ci,
    dm_test,
    newey_west_lag,
    stationary_bootstrap_indices,
)


def test_newey_west_lag_rule():
    assert newey_west_lag(100) == 4
    assert newey_west_lag(2258) == 7
    assert newey_west_lag(10) == 2


def test_dm_identical_forecasts():
    y = np.linspace(-0.02, 0.02, 200)
    pred = y + 0.001
    result = dm_test(y, pred, pred.copy())
    assert result["dm_stat"] == 0.0
    assert result["p_value"] == 1.0
    assert result["mean_loss_diff"] == 0.0


def test_dm_detects_worse_forecast_and_sign_flips():
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 0.01, size=600)
    good = y + rng.normal(0.0, 0.002, size=y.size)
    bad = y + rng.normal(0.0, 0.01, size=y.size)

    forward = dm_test(y, bad, good)
    backward = dm_test(y, good, bad)

    assert forward["dm_stat"] > 0  # bad has the larger squared errors
    assert forward["p_value"] < 0.01
    assert backward["dm_stat"] < 0
    assert np.isclose(forward["dm_stat"], -backward["dm_stat"])


def test_dm_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        dm_test(np.zeros(10), np.zeros(9), np.zeros(10))


def test_stationary_bootstrap_indices_reproducible_and_in_range():
    first = stationary_bootstrap_indices(500, avg_block=20, seed=7)
    second = stationary_bootstrap_indices(500, avg_block=20, seed=7)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (500,)
    assert first.min() >= 0 and first.max() < 500


def test_stationary_bootstrap_has_block_structure():
    idx = stationary_bootstrap_indices(2000, avg_block=20, seed=3)
    consecutive = np.mean(np.diff(idx) == 1)
    # With p = 1/20, roughly 95% of steps continue the current block.
    assert 0.9 < consecutive < 0.99


def test_bootstrap_sharpe_ci_contains_point_and_is_reproducible():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0004, 0.01, size=500)

    first = bootstrap_sharpe_ci(returns, n_boot=300, seed=123)
    second = bootstrap_sharpe_ci(returns, n_boot=300, seed=123)

    assert first["lo"] <= first["point"] <= first["hi"]
    assert first == second


def test_bootstrap_sharpe_ci_constant_returns():
    result = bootstrap_sharpe_ci(np.zeros(100), n_boot=50, seed=1)
    assert result["point"] == 0.0
    assert result["lo"] == 0.0
    assert result["hi"] == 0.0
