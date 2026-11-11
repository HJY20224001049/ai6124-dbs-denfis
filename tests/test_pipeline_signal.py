"""Implementation check: the models must learn a signal that is known to exist.

These tests are deliberately synthetic and hermetic.  They answer the question
"is the pipeline broken?" separately from "does the real DBS target contain
signal?", which is an empirical question about the data.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import rmse, zero_baseline_metrics
from src.models.baselines import XGBModel
from src.models.denfis import DENFIS

N_TRAIN = 1000
N_TEST = 500
N_FEATURES = 13


def _data(seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N_TRAIN + N_TEST, N_FEATURES))
    signal = 0.25 * X[:, 0]
    y = signal + rng.normal(0.0, 0.012, size=len(X))
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]


def _assert_beats_no_prediction(pred, y_test):
    baseline = zero_baseline_metrics(y_test)["rmse_zero"]
    model_rmse = rmse(y_test, pred)
    assert np.isfinite(pred).all()
    assert pred.shape == y_test.shape
    assert model_rmse < baseline, (
        f"model RMSE {model_rmse:.6f} did not beat no-prediction {baseline:.6f}"
    )


def test_denfis_learns_a_planted_linear_signal():
    X_train, y_train, X_test, y_test = _data()
    model = DENFIS(dthr=1.5, lam=1.0, seed=42)
    model.fit(X_train, y_train)
    _assert_beats_no_prediction(model.predict(X_test), y_test)


def test_xgboost_learns_a_planted_linear_signal():
    X_train, y_train, X_test, y_test = _data()
    model = XGBModel(
        {
            "max_depth": 2,
            "learning_rate": 0.03,
            "min_child_weight": 5,
            "reg_lambda": 1.0,
            "n_estimators": 200,
        },
        seed=42,
    )
    model.fit(X_train, y_train)
    _assert_beats_no_prediction(model.predict(X_test), y_test)
