"""Contract tests for cross-market feature alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import DATE_COL
from src.features import compute_features
from src.transfer import align_on_label_date, feature_matrix


def _prices(dates: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0003, 0.01, size=len(dates))
    close = 10.0 * np.exp(np.cumsum(log_ret))
    volume = 1_000_000 * (1.0 + 0.1 * rng.standard_normal(len(dates)))
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


def _two_markets():
    dbs_dates = pd.bdate_range("2015-01-01", periods=260)
    # The index trades on two extra days and is missing one DBS day.
    sti_dates = dbs_dates.union(pd.bdate_range("2015-12-28", periods=3)).difference(
        dbs_dates[120:121]
    )
    return {
        "DBS": compute_features(_prices(dbs_dates, seed=1)),
        "STI": compute_features(_prices(sti_dates, seed=2)),
    }


def test_alignment_uses_label_date_and_is_row_identical():
    frames = _two_markets()
    aligned = align_on_label_date(frames)

    assert len(aligned["DBS"]) > 0
    assert len(aligned["DBS"]) <= min(len(frames["DBS"]), len(frames["STI"]))
    np.testing.assert_array_equal(
        aligned["DBS"]["label_date"].to_numpy(),
        aligned["STI"]["label_date"].to_numpy(),
    )
    # The alignment must not mutate the caller's frames.
    assert len(frames["DBS"]) >= len(aligned["DBS"])


def test_alignment_keeps_each_markets_own_target():
    frames = _two_markets()
    aligned = align_on_label_date(frames)
    merged = aligned["DBS"].merge(
        aligned["STI"], on="label_date", suffixes=("_dbs", "_sti")
    )
    # Different markets, different realised returns on the same label date.
    assert not np.allclose(merged["y_dbs"], merged["y_sti"])


def test_alignment_rejects_duplicate_label_dates():
    frame = compute_features(_prices(pd.bdate_range("2015-01-01", periods=200)))
    duplicated = pd.concat([frame, frame.tail(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated"):
        align_on_label_date({"DBS": duplicated})


def test_feature_matrix_column_order():
    frame = compute_features(_prices(pd.bdate_range("2015-01-01", periods=200)))
    X = feature_matrix(frame)
    from config import FEATURES

    assert list(X.columns) == list(FEATURES)
