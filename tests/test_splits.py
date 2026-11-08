"""Contract tests for expanding walk-forward folds and purging."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.splits import fold_summary, make_folds


def _label_dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-01", "2026-10-31")


def test_fold_count_and_ids():
    folds = make_folds(_label_dates())
    assert len(folds) == 10
    assert [f.fold_id for f in folds][:3] == ["r1f1", "r1f2", "r1f3"]
    assert folds[-1].fold_id == "holdout"


def test_no_label_leakage():
    label_dates = pd.DatetimeIndex(_label_dates())
    folds = make_folds(label_dates)

    for fold in folds:
        # Test rows are exactly the labels realised inside the test window.
        test_labels = label_dates[fold.test_idx]
        assert test_labels.min() >= fold.test_start
        assert test_labels.max() <= fold.test_end

        # No training or validation label reaches the test window.
        assert label_dates[fold.train_idx].max() < fold.test_start
        assert label_dates[fold.val_idx].max() < fold.test_start
        assert label_dates[fold.select_train_idx].max() < fold.val_start

        # Validation rows are inside the validation year.
        val_labels = label_dates[fold.val_idx]
        assert val_labels.min() >= fold.val_start
        assert val_labels.max() <= fold.val_end

        # Final train = selection train + validation, by construction.
        assert len(np.intersect1d(fold.train_idx, fold.test_idx)) == 0
        assert len(np.intersect1d(fold.select_train_idx, fold.val_idx)) == 0
        assert len(fold.train_idx) == len(fold.select_train_idx) + len(fold.val_idx)


def test_summary_shape():
    label_dates = _label_dates()
    folds = make_folds(label_dates)
    summary = fold_summary(folds, label_dates)
    assert summary.shape[0] == 10
    assert summary["n_test"].min() > 200
    assert summary["n_val"].min() > 200
