"""Contract tests for the 13-feature pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COL, FEATURE_WARMUP, FEATURES, TARGET_COL
from src.features import compute_features, fit_scaler, transform_scaler


def _prices(periods: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=periods)
    log_ret = rng.normal(0.0005, 0.01, size=periods)
    close = 10.0 * np.exp(np.cumsum(log_ret))
    volume = 1_000_000 * (1.0 + 0.1 * rng.standard_normal(periods))
    return pd.DataFrame(
        {
            DATE_COL: dates,
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        }
    )


def test_feature_columns_and_labels():
    prices = _prices()
    frame = compute_features(prices)

    assert list(frame.columns) == [DATE_COL, "label_date"] + FEATURES + [TARGET_COL]
    assert len(frame) == len(prices) - FEATURE_WARMUP - 1
    assert frame[FEATURES].notna().all().all()
    assert frame[TARGET_COL].notna().all()

    lookup = prices.set_index(DATE_COL)["Adj Close"]
    for row in (0, 10, len(frame) - 1):
        date = frame.loc[row, DATE_COL]
        label_date = frame.loc[row, "label_date"]
        expected = np.log(lookup.loc[label_date] / lookup.loc[date])
        assert np.isclose(frame.loc[row, TARGET_COL], expected)


def test_features_are_causal():
    prices = _prices()
    full = compute_features(prices)

    cut = 150
    perturbed = prices.copy()
    future = perturbed.index > cut
    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        perturbed.loc[future, col] = perturbed.loc[future, col] * 2.0

    changed = compute_features(perturbed)
    cutoff_date = prices.loc[cut, DATE_COL]
    keep = full[DATE_COL] <= cutoff_date

    pd.testing.assert_frame_equal(
        full.loc[keep, FEATURES].reset_index(drop=True),
        changed.loc[keep, FEATURES].reset_index(drop=True),
        check_dtype=False,
    )


def test_scaler_uses_training_statistics_only():
    prices = _prices()
    frame = compute_features(prices)
    split = len(frame) // 2

    train = frame.iloc[:split][FEATURES]
    test = frame.iloc[split:][FEATURES]
    scaler = fit_scaler(train)

    z_train = transform_scaler(train, scaler)
    z_test = transform_scaler(test, scaler)

    assert z_train.shape == train.shape
    assert z_test.shape == test.shape
    assert np.isfinite(z_train).all()
    assert np.isfinite(z_test).all()
    assert abs(z_train.mean()) < 0.15
    assert abs(z_train.std() - 1.0) < 0.15
