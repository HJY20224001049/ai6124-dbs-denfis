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


def test_all_folds_excludes_holdout_and_rejects_unknown(monkeypatch):
    import pytest
    from src.evaluation import cv

    frame = _frame()
    folds = make_folds_from_frame(frame)
    seen = []

    def run(model_name, factory, fold, *args, **kwargs):
        seen.append(fold.fold_id)
        return pd.DataFrame(), {"rmse": 0., "da": 0., "sharpe": 0., "tau": 0.}

    monkeypatch.setattr(cv, "run_fold", run)
    monkeypatch.setattr(cv, "save_fold_result", lambda *args: None)
    monkeypatch.setattr(cv, "build_summary", lambda: pd.DataFrame())
    monkeypatch.setattr(cv, "write_manifest", lambda: None)
    cv.run_model_on_folds("dummy", None, folds, frame, [{}])
    assert len(seen) == 9 and "holdout" not in seen
    seen.clear()
    cv.run_model_on_folds("dummy", None, folds, frame, [{}], fold_ids=["holdout"])
    assert seen == ["holdout"]
    with pytest.raises(ValueError, match="unknown fold"):
        cv.run_model_on_folds("dummy", None, folds, frame, [{}], fold_ids=["typo"])


def test_online_runner_predicts_before_realised_label_update():
    frame = _frame()
    fold = get_fold(make_folds_from_frame(frame), "r1f1")
    created = []

    class RecordingModel(DummyModel):
        def __init__(self):
            super().__init__()
            self.events = []

        def predict(self, X):
            self.events.append(("predict", len(X)))
            return super().predict(X)

        def partial_fit(self, x, y):
            self.events.append(("update", float(y)))

    def factory(params, seed):
        model = RecordingModel()
        created.append(model)
        return model

    run_fold("online_dummy", factory, fold, frame, [{}], seed=42, online=True)
    events = created[-1].events
    assert len(events) == 2 * len(fold.test_idx)
    for i, target in enumerate(frame.iloc[fold.test_idx][TARGET_COL]):
        assert events[2*i] == ("predict", 1)
        assert events[2*i+1] == ("update", float(target))


def test_equal_validation_rmse_prefers_fewer_rules_then_larger_threshold():
    frame = _frame()
    fold = get_fold(make_folds_from_frame(frame), "r1f1")

    def factory(params, seed):
        model = DummyModel(params, seed)
        model.n_rules_ = params["rules"]
        return model

    grid = [{"rules": 10, "dthr": 1.5}, {"rules": 2, "dthr": .25},
            {"rules": 2, "dthr": 1.0}]
    _, metrics = run_fold("tie", factory, fold, frame, grid, seed=42)
    assert metrics["params"] == {"rules": 2, "dthr": 1.0}


def test_source_only_selection_does_not_use_target_validation_labels():
    frame = _frame()
    source = frame.copy()
    source[TARGET_COL] = .001
    frame[TARGET_COL] = -.01
    fold = get_fold(make_folds_from_frame(frame), "r1f1")

    class FixedModel:
        def __init__(self, value):
            self.value = value

        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.full(len(X), self.value)

    grid = [{"value": .001}, {"value": -.01}]
    factory = lambda p, seed: FixedModel(p["value"])
    _, source_metrics = run_fold("zero_shot", factory, fold, frame, grid,
                                 seed=42, selection_frame=source)
    _, target_metrics = run_fold("scratch", factory, fold, frame, grid, seed=42)
    assert source_metrics["params"] == {"value": .001}
    assert target_metrics["params"] == {"value": -.01}
