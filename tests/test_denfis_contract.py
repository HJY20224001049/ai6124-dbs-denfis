"""DENFIS contracts: learning, ECM geometry, repeatability and pure inference."""

import copy

import numpy as np
import pytest

from src.models.denfis import DENFIS


def _data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(500, 2))
    return X, np.sin(X[:, 0]) + 0.5 * X[:, 1]


def _assert_state_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], np.ndarray):
            np.testing.assert_array_equal(a[key], b[key])
        else:
            assert a[key] == b[key]


def test_nonlinear_generalisation_and_reset():
    X, y = _data()
    model = DENFIS(dthr=0.4, lam=1.0, prediction_clip=None)
    model.fit(X[:400], y[:400])
    before = copy.deepcopy(model.__dict__)
    pred = model.predict(X[400:])
    assert pred.shape == (100,) and np.isfinite(pred).all()
    assert np.mean((pred-y[400:])**2) < 0.3 * np.mean(y[400:]**2)
    assert 2 <= model.n_rules_ <= 50
    _assert_state_equal(before, model.__dict__)
    model.fit(X[:400], y[:400])
    np.testing.assert_array_equal(pred, model.predict(X[400:]))
    other = DENFIS(dthr=0.4, lam=1.0, prediction_clip=None)
    other.fit(X[:400], y[:400])
    np.testing.assert_array_equal(pred, other.predict(X[400:]))


def test_ecm_geometry_and_rule_cap():
    model = DENFIS(dthr=1, merge_factor=0, max_rules=2)
    model.fit(np.array([[0.], [1.], [10.], [30.]]), np.zeros(4))
    assert model.n_rules_ == 2 and model.cap_hits_ == 1
    assert model.centers_[0, 0] == pytest.approx(0.5)
    assert model.radii_[0] == pytest.approx(0.5)
    X, y = _data()
    counts = []
    for threshold in (0.25, 1.0):
        m = DENFIS(dthr=threshold, merge_factor=0)
        m.fit(X, y)
        counts.append(m.n_rules_)
    assert counts[0] > counts[1]


def test_online_update_and_frozen_antecedents():
    X, y = _data()
    model = DENFIS(dthr=0.3, evolve_rules=False)
    model.fit(X[:100], y[:100] * 0.01)
    centers = model.centers_.copy()
    coef = model.coef_.copy()
    model.partial_fit(np.array([8., 8.]), 0.01)
    np.testing.assert_array_equal(model.centers_, centers)
    assert not np.array_equal(model.coef_, coef)
    model.evolve_rules = True
    count = model.n_rules_
    model.partial_fit(np.array([20., 20.]), 0.01)
    assert model.n_rules_ == count + 1


def test_forgetting_and_merge_are_effective():
    X, y = _data()
    a, b = DENFIS(lam=1), DENFIS(lam=0.95)
    for m in (a, b):
        m.fit(X[:100], y[:100] * 0.01)
    assert not np.allclose(a.predict(X[100:]), b.predict(X[100:]))
    # Two clusters start far apart then move within the merge threshold.
    m = DENFIS(dthr=1, merge_factor=0.5)
    m.fit(np.array([[0.], [3.]]), np.zeros(2))
    m.centers_[1] = 0.1
    m._merge_rules()
    assert m.n_rules_ == 1 and m.merges_ == 1


def test_degenerate_inputs_and_validation():
    model = DENFIS()
    with pytest.raises(RuntimeError):
        model.predict(np.ones((2, 13)))
    model.fit(np.zeros((20, 13)), np.zeros(20))
    assert model.n_rules_ == 1
    np.testing.assert_array_equal(model.predict(np.zeros((3, 13))), np.zeros(3))
    assert model.predict(np.empty((0, 13))).shape == (0,)
    with pytest.raises(ValueError):
        model.predict(np.ones((2, 12)))
    with pytest.raises(ValueError):
        model.partial_fit(np.zeros(13), np.nan)
    with pytest.raises(ValueError):
        model.fit(np.zeros((0, 13)), np.empty(0))
    for kwargs in ({"lam": 0}, {"dthr": np.nan}, {"max_rules": 0}):
        with pytest.raises(ValueError):
            DENFIS(**kwargs)


def test_one_rule_rls_matches_regularised_least_squares():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(80, 3))
    y = .01 + X @ np.array([.02, -.01, .03])
    model = DENFIS(max_rules=1, lam=1, prediction_clip=None)
    model.fit(X, y)
    phi = np.c_[np.ones(len(X)), X]
    expected = np.linalg.solve(phi.T @ phi + .01*np.eye(4), phi.T @ y)
    np.testing.assert_allclose(model.coef_[0], expected, atol=1e-12)
    assert np.linalg.eigvalsh(model.cov_[0]).min() > 0


def test_incremental_merge_screen_matches_full_pairwise_reference():
    class FullScan(DENFIS):
        def partial_fit(self, x, y):
            self._ecm_update(x)
            self._merge_rules()
            self._update_consequents(x, y)
            self.rule_history_.append(self.n_rules_)

    X, y = _data()
    a = DENFIS(dthr=.4, evolve_rules=True, max_rules=15)
    b = FullScan(dthr=.4, evolve_rules=True, max_rules=15)
    for model in (a, b):
        model.fit(X[:100], .01*y[:100])
        for x, target in zip(X[100:], .01*y[100:]):
            model.partial_fit(x, target)
    _assert_state_equal(a.__dict__, b.__dict__)
