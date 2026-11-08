"""Expanding walk-forward folds with label-date purging.

Each row's label is the next trading day's return, so folds are assigned by
``label_date`` rather than by the feature date.  This guarantees that a test
year contains exactly the returns realised inside that year, and that no
training label ever reaches into a validation or test period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import FOLD_SPECS, HOLDOUT_SPEC


@dataclass(frozen=True)
class Fold:
    fold_id: str
    rep: int
    fold: int
    test_year: int | None
    select_train_idx: np.ndarray
    val_idx: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    select_train_end: pd.Timestamp
    final_train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _idx(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask, dtype=bool))


def _windows_for_year(test_year: int) -> dict[str, pd.Timestamp]:
    return {
        "select_train_end": pd.Timestamp(f"{test_year - 2}-12-31"),
        "val_start": pd.Timestamp(f"{test_year - 1}-01-01"),
        "val_end": pd.Timestamp(f"{test_year - 1}-12-31"),
        "final_train_end": pd.Timestamp(f"{test_year - 1}-12-31"),
        "test_start": pd.Timestamp(f"{test_year}-01-01"),
        "test_end": pd.Timestamp(f"{test_year}-12-31"),
    }


def _build_fold(
    label_dates: pd.DatetimeIndex,
    *,
    fold_id: str,
    rep: int,
    fold: int,
    test_year: int | None,
    windows: dict[str, pd.Timestamp],
) -> Fold:
    select_train_idx = _idx(label_dates <= windows["select_train_end"])
    val_idx = _idx(
        (label_dates >= windows["val_start"]) & (label_dates <= windows["val_end"])
    )
    train_idx = _idx(label_dates <= windows["final_train_end"])
    test_idx = _idx(
        (label_dates >= windows["test_start"]) & (label_dates <= windows["test_end"])
    )

    for name, idx in [
        ("select_train", select_train_idx),
        ("val", val_idx),
        ("train", train_idx),
        ("test", test_idx),
    ]:
        if len(idx) == 0:
            raise ValueError(f"{fold_id}: empty {name} index set")

    # Structural guarantees: no training label reaches validation or test.
    if label_dates[select_train_idx].max() >= windows["val_start"]:
        raise AssertionError(f"{fold_id}: selection train leaks into validation")
    if label_dates[val_idx].max() >= windows["test_start"]:
        raise AssertionError(f"{fold_id}: validation leaks into test")
    if label_dates[train_idx].max() >= windows["test_start"]:
        raise AssertionError(f"{fold_id}: final train leaks into test")
    if len(np.intersect1d(train_idx, test_idx)):
        raise AssertionError(f"{fold_id}: train/test overlap")

    return Fold(
        fold_id=fold_id,
        rep=rep,
        fold=fold,
        test_year=test_year,
        select_train_idx=select_train_idx,
        val_idx=val_idx,
        train_idx=train_idx,
        test_idx=test_idx,
        **windows,
    )


def make_folds(
    label_dates,
    specs: list[dict] | None = None,
    holdout: dict | None = None,
) -> list[Fold]:
    """Build nine regular folds plus the 2026 holdout."""
    specs = FOLD_SPECS if specs is None else specs
    holdout = HOLDOUT_SPEC if holdout is None else holdout
    label_dates = pd.DatetimeIndex(pd.to_datetime(label_dates))

    folds: list[Fold] = []
    for spec in specs:
        windows = _windows_for_year(int(spec["test_year"]))
        folds.append(
            _build_fold(
                label_dates,
                fold_id=str(spec["fold_id"]),
                rep=int(spec["rep"]),
                fold=int(spec["fold"]),
                test_year=int(spec["test_year"]),
                windows=windows,
            )
        )

    holdout_windows = {
        "select_train_end": pd.Timestamp(holdout["select_train_end"]),
        "val_start": pd.Timestamp(holdout["val_start"]),
        "val_end": pd.Timestamp(holdout["val_end"]),
        "final_train_end": pd.Timestamp(holdout["final_train_end"]),
        "test_start": pd.Timestamp(holdout["test_start"]),
        "test_end": pd.Timestamp(holdout["test_end"]),
    }
    folds.append(
        _build_fold(
            label_dates,
            fold_id=str(holdout["fold_id"]),
            rep=int(holdout["rep"]),
            fold=int(holdout["fold"]),
            test_year=None,
            windows=holdout_windows,
        )
    )
    return folds


def make_folds_from_frame(frame: pd.DataFrame) -> list[Fold]:
    """Convenience wrapper for a feature frame produced by compute_features."""
    return make_folds(pd.to_datetime(frame["label_date"]))


def fold_summary(folds: list[Fold], label_dates) -> pd.DataFrame:
    """Human-readable table used in the notebook and in tests."""
    label_dates = pd.DatetimeIndex(pd.to_datetime(label_dates))
    rows = []
    for fold in folds:
        rows.append(
            {
                "fold_id": fold.fold_id,
                "rep": fold.rep,
                "fold": fold.fold,
                "test_year": fold.test_year,
                "n_select_train": len(fold.select_train_idx),
                "n_val": len(fold.val_idx),
                "n_train": len(fold.train_idx),
                "n_test": len(fold.test_idx),
                "first_test_label": label_dates[fold.test_idx].min().date(),
                "last_test_label": label_dates[fold.test_idx].max().date(),
            }
        )
    return pd.DataFrame(rows)


def get_fold(folds: list[Fold], fold_id: str) -> Fold:
    for fold in folds:
        if fold.fold_id == fold_id:
            return fold
    raise KeyError(f"Unknown fold_id: {fold_id}")
