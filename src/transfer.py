"""Feature alignment helpers for the ^STI → DBS transfer experiment.

The transfer protocols (scratch / zero-shot / evolve) compare models across
two markets, so the two feature frames must be aligned on the day whose return
is being predicted (``label_date``), never on the feature date.  Aligning on
``label_date`` guarantees that a row used on one market is matched to a row
whose target is realised on the same trading day on the other market.

Each market is standardised separately with :func:`src.features.fit_scaler`
on that market's training rows.  DENFIS rules are therefore transferred in
standardised space; the report must state this assumption.

Note on ^STI: index volume is frequently zero by construction, so ``vol_z``
is degenerate (mostly zero) for the index.  The main experiment still uses the
same 13 columns, but the transfer write-up should not interpret ``vol_z`` as
informative for the index leg.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from config import DATE_COL, FEATURES
from src.data import load_prices
from src.features import compute_features

MARKET_FILES = {"DBS": "D05_SI", "STI": "STI"}


def load_market_frame(market: str) -> pd.DataFrame:
    """Load a cached market and build the protocol feature frame."""
    try:
        name = MARKET_FILES[market]
    except KeyError as exc:
        raise ValueError(
            f"unknown market {market!r}; expected one of {sorted(MARKET_FILES)}"
        ) from exc
    return compute_features(load_prices(name))


def align_on_label_date(
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Restrict every frame to the ``label_date`` values shared by all markets.

    Returns fresh, row-aligned copies sorted by ``label_date``.  The input
    frames are not modified.
    """
    if not frames:
        raise ValueError("need at least one frame")
    for market, frame in frames.items():
        missing = set([DATE_COL, "label_date", *FEATURES]) - set(frame.columns)
        if missing:
            raise ValueError(f"{market}: missing columns {sorted(missing)}")
        if frame["label_date"].duplicated().any():
            raise ValueError(f"{market}: duplicated label_date")

    common = None
    for frame in frames.values():
        dates = set(pd.to_datetime(frame["label_date"]))
        common = dates if common is None else common & dates
    if not common:
        raise ValueError("markets have no label_date in common")

    common_idx = pd.DatetimeIndex(sorted(common))
    aligned: dict[str, pd.DataFrame] = {}
    for market, frame in frames.items():
        out = frame.copy()
        out["label_date"] = pd.to_datetime(out["label_date"])
        out = out[out["label_date"].isin(common_idx)]
        out = out.sort_values("label_date").reset_index(drop=True)
        if len(out) != len(common_idx):
            raise ValueError(f"{market}: alignment lost rows")
        aligned[market] = out
    return aligned


def load_transfer_frames() -> dict[str, pd.DataFrame]:
    """Load DBS and ^STI, build both feature frames, and align them."""
    frames = {market: load_market_frame(market) for market in MARKET_FILES}
    aligned = align_on_label_date(frames)
    # The row order is now identical across markets; assert it explicitly so a
    # future data update cannot silently misalign the transfer experiment.
    dates = [frame["label_date"].to_numpy() for frame in aligned.values()]
    if any(not (dates[0] == other).all() for other in dates[1:]):
        raise ValueError("aligned markets are not row-identical")
    return aligned


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """The 13 protocol features in frozen column order."""
    return frame[list(FEATURES)]
