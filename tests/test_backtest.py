"""Unit tests for the long/flat backtest and threshold selection."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.backtest import (
    backtest_metrics,
    log_to_simple,
    make_signal,
    max_drawdown,
    run_backtest,
    select_tau,
    sharpe,
)

RETURNS = np.array([0.01, -0.02, 0.03, 0.0])
SIGNAL = np.array([1, 1, 0, 1])


def test_turnover_and_returns_with_cost():
    bt = run_backtest(SIGNAL, RETURNS, cost_bps=10)
    assert bt["turnover"].tolist() == [1.0, 0.0, 1.0, 1.0]
    expected = SIGNAL * RETURNS - 0.001 * np.array([1.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(bt["strat_ret"].to_numpy(), expected)


def test_all_long_with_zero_cost_matches_benchmark():
    bt = run_backtest(np.ones_like(SIGNAL), RETURNS, cost_bps=0)
    np.testing.assert_allclose(bt["strat_ret"].to_numpy(), RETURNS)


def test_all_flat_has_zero_return():
    bt = run_backtest(np.zeros_like(SIGNAL), RETURNS, cost_bps=10)
    np.testing.assert_allclose(bt["strat_ret"].to_numpy(), np.zeros_like(RETURNS))


def test_max_drawdown():
    assert np.isclose(max_drawdown(np.array([0.1, -0.2, 0.05])), 0.2)


def test_make_signal():
    pred = np.array([0.0001, -0.0001, 0.002, 0.0006])
    assert make_signal(pred, 0.0).tolist() == [1, 0, 1, 1]
    assert make_signal(pred, 0.001).tolist() == [0, 0, 1, 0]


def test_metrics_keys():
    bt = run_backtest(SIGNAL, RETURNS)
    metrics = backtest_metrics(bt["strat_ret"], bt["signal"])
    assert set(metrics) == {
        "n",
        "sharpe",
        "ann_ret",
        "max_dd",
        "turnover",
        "exposure",
        "cum_return",
    }
    assert metrics["exposure"] == pytest.approx(0.75)


def test_select_tau_tie_goes_to_smaller_threshold():
    y_true = np.array([0.01, 0.02, 0.01, 0.02])
    y_pred = np.array([0.005, 0.015, 0.005, 0.015])
    # Both tau=0 and tau=5bp produce the same all-long position.
    assert select_tau(y_true, y_pred) == 0.0


def test_log_to_simple():
    np.testing.assert_allclose(log_to_simple(np.array([0.0, np.log(1.1)])), [0.0, 0.1])


def test_sharpe_constant_returns_is_zero():
    assert sharpe(np.zeros(10)) == 0.0
