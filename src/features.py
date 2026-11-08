"""Causal feature engineering for the 13 frozen protocol features.

The output frame carries ``Date`` (feature date), ``label_date`` (the trading
day whose return is predicted), the 13 features, and ``y`` (next-day log
return).  Folds are assigned by ``label_date`` so that a test year contains
returns realised inside that year, not features stamped inside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    DATE_COL,
    FEATURE_WARMUP,
    FEATURES,
    TARGET_COL,
)


def _rsi(price: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI, returning values in [0, 100]."""
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), 50.0)
    return rsi


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build the protocol's 13 features and next-day log-return label.

    Parameters
    ----------
    prices : DataFrame
        Cleaned daily bars with Date, Open, High, Low, Close, Adj Close, Volume.
    """
    out = prices.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out = out.sort_values(DATE_COL).drop_duplicates(DATE_COL, keep="last")
    out = out.reset_index(drop=True)

    price = out["Adj Close"].astype(float)
    volume = out["Volume"].astype(float)
    ret = np.log(price / price.shift(1))

    out["ret_0"] = ret
    for lag in range(1, 5):
        out[f"ret_{lag}"] = ret.shift(lag)

    for window in (5, 20, 60):
        out[f"ma{window}"] = price / price.rolling(window).mean() - 1.0

    out["rsi14"] = (_rsi(price) - 50.0) / 50.0

    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    out["macd"] = (macd_line - signal_line) / price

    ma20 = price.rolling(20).mean()
    sd20 = price.rolling(20).std(ddof=0)
    upper = ma20 + 2.0 * sd20
    lower = ma20 - 2.0 * sd20
    width = upper - lower
    boll = (price - lower) / width
    out["boll_b"] = boll.where(width.abs() > 1e-12, 0.5)

    out["rv20"] = ret.rolling(20).std(ddof=1) * np.sqrt(252.0)

    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std(ddof=1)
    vol_z = (volume - vol_mean) / vol_std.replace(0.0, np.nan)
    out["vol_z"] = vol_z.fillna(0.0)

    out[TARGET_COL] = ret.shift(-1)
    out["label_date"] = out[DATE_COL].shift(-1)

    # Drop the warmup block, then any row that is still undefined.
    out = out.iloc[FEATURE_WARMUP:].copy()
    out = out.dropna(subset=FEATURES + [TARGET_COL, "label_date"])

    return out[[DATE_COL, "label_date"] + FEATURES + [TARGET_COL]].reset_index(drop=True)


def to_xy(frame: pd.DataFrame):
    """Return positional arrays plus date indexers from a feature frame."""
    X = frame[FEATURES].to_numpy(dtype=np.float64)
    y = frame[TARGET_COL].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(frame[DATE_COL])
    label_dates = pd.to_datetime(frame["label_date"])
    return X, y, dates, label_dates


@dataclass(frozen=True)
class FeatureScaler:
    """Per-fold winsorisation and standardisation parameters."""

    lower: pd.Series
    upper: pd.Series
    mean: pd.Series
    std: pd.Series


def fit_scaler(
    X: pd.DataFrame,
    lower_q: float = 0.005,
    upper_q: float = 0.995,
) -> FeatureScaler:
    """Fit winsorisation bounds and z-score statistics on training rows only."""
    lower = X.quantile(lower_q)
    upper = X.quantile(upper_q)
    clipped = X.clip(lower, upper, axis=1)
    mean = clipped.mean()
    std = clipped.std(ddof=1).replace(0.0, 1.0)
    return FeatureScaler(lower=lower, upper=upper, mean=mean, std=std)


def transform_scaler(X: pd.DataFrame, scaler: FeatureScaler) -> np.ndarray:
    """Apply a fitted scaler and return a float64 array in FEATURES order."""
    clipped = X.clip(scaler.lower, scaler.upper, axis=1)
    scaled = (clipped - scaler.mean) / scaler.std
    return scaled.to_numpy(dtype=np.float64)


def scale_fold(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    other_indices: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], FeatureScaler]:
    """Fit on train_idx and transform every requested index set."""
    train_X = frame.iloc[train_idx][FEATURES]
    scaler = fit_scaler(train_X)
    transformed = {
        name: transform_scaler(frame.iloc[idx][FEATURES], scaler)
        for name, idx in other_indices.items()
    }
    return transformed, scaler
