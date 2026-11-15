"""Contracts for the B2 equal-weight one-rule DENFIS ensemble."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.denfis import DENFIS
from src.models.ensemble import DENFISEnsemble


def _data():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 5))
    y = 0.02 * X[:, 0] - 0.01 * X[:, 1] + 0.015 * X[:, 2] + rng.normal(scale=0.01, size=300)
    return X, y


def test_ensemble_is_the_mean_of_its_components():
    X, y = _data()
    components = [[0, 3, 4], [1, 3, 4], [2, 3, 4]]
    model = DENFISEnsemble(components, dthr=3.0, lam=1.0, seed=7)
    model.fit(X[:200], y[:200])
    pred = model.predict(X[200:])

    parts = []
    for index, comp in enumerate(components):
        single = DENFIS(dthr=3.0, lam=1.0, max_rules=1, seed=7 + index)
        single.fit(X[:200][:, comp], y[:200])
        parts.append(single.predict(X[200:][:, comp]))
    np.testing.assert_allclose(pred, np.mean(parts, axis=0), atol=1e-12)
    assert model.n_rules_ == 3
    assert np.isfinite(pred).all()


def test_ensemble_rejects_bad_components():
    with pytest.raises(ValueError):
        DENFISEnsemble([])
    with pytest.raises(ValueError):
        DENFISEnsemble([[]])
    model = DENFISEnsemble([[0, 1]])
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((2, 2)))


def test_ensemble_ties_break_on_rule_count_not_a_single_flag():
    model = DENFISEnsemble([[0], [1]], dthr=3.0, lam=1.0)
    X, y = _data()
    model.fit(X[:100], y[:100])
    assert model.get_params()["components"] == [[0], [1]]
    assert model.n_rules_ == 2
