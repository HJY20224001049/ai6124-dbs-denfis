"""Hand-checked unit tests for metrics."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    compute_metrics,
    directional_accuracy,
    information_coefficient,
    mae,
    r2,
    rmse,
)

Y_TRUE = np.array([0.01, -0.02, 0.03, -0.04])
Y_PRED = np.array([0.005, -0.01, -0.02, 0.01])


def test_rmse_mae():
    assert np.isclose(rmse(Y_TRUE, Y_PRED), 0.03579427202847139)
    assert np.isclose(mae(Y_TRUE, Y_PRED), 0.02875)


def test_r2():
    sse = np.sum((Y_TRUE - Y_PRED) ** 2)
    sst = np.sum((Y_TRUE - Y_TRUE.mean()) ** 2)
    assert np.isclose(r2(Y_TRUE, Y_PRED), 1.0 - sse / sst)


def test_directional_accuracy():
    assert np.isclose(directional_accuracy(Y_TRUE, Y_PRED), 0.5)
    # All four absolute returns exceed 5 bp, so the deadband version matches.
    assert np.isclose(directional_accuracy(Y_TRUE, Y_PRED, deadband=5e-4), 0.5)


def test_information_coefficient():
    assert np.isclose(information_coefficient(Y_TRUE, Y_PRED), np.corrcoef(Y_TRUE, Y_PRED)[0, 1])


def test_compute_metrics_keys():
    metrics = compute_metrics(Y_TRUE, Y_PRED)
    assert set(metrics) == {"n", "rmse", "mae", "r2", "ic", "da", "da5"}
    assert metrics["n"] == 4


def test_constant_prediction_has_zero_ic():
    assert information_coefficient(Y_TRUE, np.zeros_like(Y_TRUE)) == 0.0
