"""Contract test for the CV runner using a deterministic dummy model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COL, FEATURES, TARGET_COL
from src.evaluation.cv import build_sequences, run_fold
from src.splits import get_fold, make_folds_from_frame


class DummyModel:
    name = "dummy"

    def __init__(self, params=None, seed=42):
        self.params = dict(params or {})
        self.seed = seed
        self.mean_ = 0.0

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))

    def predict(self, X):
        return np.full(len(X), self.mean_, dtype=np.float64)


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    label_dates = pd.bdate_range("2010-01-01", "2026-10-31")
    n = len(label_dates)
    data = {
        DATE_COL: label_dates - pd.Timedelta(days=1),
        "label_date": label_dates,
        TARGET_COL: rng.normal(0.0, 0.01, size=n),
    }
    for feature in FEATURES:
        data[feature] = rng.normal(size=n)
    return pd.DataFrame(data)


class DummySeqModel(DummyModel):
    name = "dummy_seq"
    requires_sequence = True
    seq_len = 20

    def fit(self, X, y):
        assert X.ndim == 3
        self.mean_ = float(np.mean(y))

    def predict(self, X):
        assert X.ndim == 3
        return np.full(len(X), self.mean_, dtype=np.float64)


def test_build_sequences_is_causal_and_padded():
    full = np.arange(40 * 3, dtype=float).reshape(40, 3)
    sequences, valid = build_sequences(full, np.array([0, 5, 25, 30]), seq_len=10)
    assert valid.tolist() == [25, 30]
    assert sequences.shape == (2, 10, 3)
    np.testing.assert_array_equal(sequences[0, -1], full[25])
    np.testing.assert_array_equal(sequences[1, -1], full[30])


def test_run_fold_contract():
    frame = _frame()
    folds = make_folds_from_frame(frame)
    fold = get_fold(folds, "r1f1")

    pred, metrics = run_fold(
        "dummy",
        lambda params, seed: DummyModel(params, seed),
        fold,
        frame,
        [{}],
        seed=42,
    )

    assert len(pred) == len(fold.test_idx)
    assert list(pred.columns) == [
        "date",
        "y_true",
        "y_pred",
        "signal",
        "position",
        "strat_ret",
        "bench_ret",
    ]
    assert pred.notna().all().all()
    assert set(pred["signal"].unique()).issubset({0, 1})
    assert metrics["model"] == "dummy"
    assert metrics["fold_id"] == "r1f1"
    assert metrics["n"] == len(fold.test_idx)
    assert metrics["tau"] in (0.0, 5e-4, 1e-3, 2e-3)


def test_run_fold_sequence_contract():
    frame = _frame()
    folds = make_folds_from_frame(frame)
    fold = get_fold(folds, "r1f1")

    pred, metrics = run_fold(
        "dummy_seq",
        lambda params, seed: DummySeqModel(params, seed),
        fold,
        frame,
        [{}],
        seed=42,
        sequence_len=20,
    )

    assert len(pred) == len(fold.test_idx)
    assert metrics["model"] == "dummy_seq"
    assert metrics["fold_id"] == "r1f1"
    assert pred.notna().all().all()
