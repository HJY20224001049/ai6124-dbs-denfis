"""Bounded ECM / Gaussian TSK / local weighted-RLS research implementation.

See docs/denfis_notes.md for formulas and differences from original DENFIS.
Inputs must already be standardised; no dates, prices or files are accessed.
"""

from __future__ import annotations

import numpy as np

from config import DENFIS_MAX_RULES, DENFIS_MERGE_FACTOR, GLOBAL_SEED


class DENFIS:
    """First-order TSK rules learned with ECM and chronological weighted RLS.

    ``fit`` first learns antecedents from training X, then replays training
    pairs once to estimate consequents. ``partial_fit`` never resets state.
    ``evolve_rules`` controls antecedent adaptation during partial updates,
    not whether consequents may adapt. Predictions are pure reads.
    """

    name = "denfis"

    def __init__(
        self, dthr: float = 0.5, lam: float = 0.99,
        max_rules: int = DENFIS_MAX_RULES,
        merge_factor: float = DENFIS_MERGE_FACTOR,
        seed: int = GLOBAL_SEED, evolve_rules: bool = False,
        *, prediction_clip: float | None = 0.2, rls: bool = True,
    ):
        if not np.isfinite(dthr) or dthr <= 0:
            raise ValueError("dthr must be positive and finite")
        if not np.isfinite(lam) or not 0 < lam <= 1:
            raise ValueError("lam must be in (0, 1]")
        if int(max_rules) != max_rules or max_rules < 1:
            raise ValueError("max_rules must be a positive integer")
        if not np.isfinite(merge_factor) or not 0 <= merge_factor <= 1:
            raise ValueError("merge_factor must be in [0, 1]")
        if prediction_clip is not None and (
            not np.isfinite(prediction_clip) or prediction_clip <= 0
        ):
            raise ValueError("prediction_clip must be positive or None")
        self.dthr, self.lam = float(dthr), float(lam)
        self.max_rules, self.merge_factor = int(max_rules), float(merge_factor)
        self.seed, self.evolve_rules = int(seed), bool(evolve_rules)
        self.prediction_clip, self.rls = prediction_clip, bool(rls)

    @property
    def n_rules_(self) -> int:
        return len(self.centers_) if hasattr(self, "centers_") else 0

    def get_params(self) -> dict:
        return {key: getattr(self, key) for key in (
            "dthr", "lam", "max_rules", "merge_factor", "seed",
            "evolve_rules", "prediction_clip", "rls",
        )}

    def _inputs(self, X: np.ndarray, *, training: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError("X must be a finite (n, d) matrix with d > 0")
        if training and len(X) == 0:
            raise ValueError("empty training set")
        if not training:
            if not self.n_rules_:
                raise RuntimeError("fit must be called before prediction/update")
            if X.shape[1] != self.n_features_in_:
                raise ValueError("feature dimension mismatch")
        return X

    def _new_rule(self, x: np.ndarray) -> None:
        self.centers_ = np.vstack((self.centers_, x))
        self.radii_ = np.append(self.radii_, 0.0)
        self.support_ = np.append(self.support_, 1)
        p = self.n_features_in_ + 1
        self.coef_ = np.vstack((self.coef_, np.zeros(p)))
        self.cov_ = np.concatenate((self.cov_, (100.0 * np.eye(p))[None]))
        self.mass_ = np.append(self.mass_, 0.0)
        self.target_sum_ = np.append(self.target_sum_, 0.0)

    def _ecm_update(self, x: np.ndarray) -> None:
        if not self.n_rules_:
            self._new_rule(x)
            return
        distances = np.sqrt(np.mean((self.centers_ - x) ** 2, axis=1))
        inside = np.flatnonzero(distances <= self.radii_ + 1e-12)
        if len(inside):
            index = int(inside[np.argmin(distances[inside])])
        else:
            extended = distances + self.radii_
            index = int(np.argmin(extended))
            if extended[index] > 2 * self.dthr:
                if self.n_rules_ < self.max_rules:
                    self._new_rule(x)
                    return
                # Bounded-memory extension: preserve antecedents at the cap.
                index = int(np.argmin(distances))
                self.cap_hits_ += 1
            else:
                radius = extended[index] / 2
                shift = (distances[index] - radius) / max(distances[index], 1e-12)
                self.centers_[index] += shift * (x - self.centers_[index])
                self.radii_[index] = radius
        self.support_[index] += 1

    def _merge_rules(self) -> None:
        if self.merge_factor == 0:
            return
        while self.n_rules_ > 1:
            distance = np.sqrt(np.mean(
                (self.centers_[:, None] - self.centers_[None, :]) ** 2, axis=2
            ))
            np.fill_diagonal(distance, np.inf)
            i, j = np.unravel_index(np.argmin(distance), distance.shape)
            if distance[i, j] >= self.merge_factor * self.dthr:
                break
            if self.support_[j] > self.support_[i]:
                i, j = j, i
            # Keep the better-supported antecedent and its local fit. Enlarge
            # radius to enclose the removed ball; merge is a project extension.
            self.radii_[i] = max(self.radii_[i], distance[i, j] + self.radii_[j])
            self.support_[i] += self.support_[j]
            for attr in ("centers_", "radii_", "support_", "coef_", "cov_", "mass_", "target_sum_"):
                setattr(self, attr, np.delete(getattr(self, attr), j, axis=0))
            self.merges_ += 1

    def firing_strengths(self, X: np.ndarray) -> np.ndarray:
        """Return normalised Gaussian activations without modifying state."""
        X = self._inputs(X)
        width = np.maximum(self.radii_, max(0.1 * self.dthr, 1e-6))
        log_weight = -0.5 * np.mean(
            (X[:, None, :] - self.centers_[None, :, :]) ** 2, axis=2
        ) / width[None, :] ** 2
        log_weight -= np.max(log_weight, axis=1, keepdims=True)
        weights = np.exp(log_weight)
        return weights / weights.sum(axis=1, keepdims=True)

    def _update_consequents(self, x: np.ndarray, y: float) -> None:
        h = self.firing_strengths(x[None])[0]
        phi = np.r_[1.0, x]
        if not self.rls:
            self.mass_ += h
            self.target_sum_ += h * y
            self.coef_[:, 0] = self.target_sum_ / np.maximum(self.mass_, 1e-12)
            return
        # Tiny activations do not discount unused rules or inflate covariance.
        active = h >= 1e-6
        P = self.cov_[active]
        v = np.einsum("kij,j->ki", P, phi)
        denominator = self.lam + h[active] * (v @ phi)
        gain = h[active, None] * v / denominator[:, None]
        error = y - self.coef_[active] @ phi
        self.coef_[active] += gain * error[:, None]
        updated = (P - gain[:, :, None] * v[:, None, :]) / self.lam
        updated = 0.5 * (updated + updated.transpose(0, 2, 1))
        bad = (~np.isfinite(updated).all(axis=(1, 2))
               | (np.trace(updated, axis1=1, axis2=2) > 1e8)
               | (np.diagonal(updated, axis1=1, axis2=2).min(axis=1) <= 0)
               | ~np.isfinite(self.coef_[active]).all(axis=1))
        if bad.any():
            updated[bad] = 100.0 * np.eye(len(phi))
            indices = np.flatnonzero(active)[bad]
            self.coef_[indices] = 0.0
            self.resets_ += int(bad.sum())
        self.cov_[active] = updated

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Reset; learn ECM antecedents and replay training labels once."""
        X = self._inputs(X, training=True)
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (len(X),) or not np.isfinite(y).all():
            raise ValueError("y must be finite with shape (n,)")
        self.n_features_in_ = X.shape[1]
        p = X.shape[1] + 1
        self.centers_, self.radii_ = np.empty((0, X.shape[1])), np.empty(0)
        self.support_ = np.empty(0, dtype=int)
        self.coef_, self.cov_ = np.empty((0, p)), np.empty((0, p, p))
        self.mass_, self.target_sum_ = np.empty(0), np.empty(0)
        self.cap_hits_ = self.merges_ = self.resets_ = 0
        for x in X:
            self._ecm_update(x)
        self._merge_rules()
        for x, target in zip(X, y):
            self._update_consequents(x, float(target))
        self.rule_history_ = [self.n_rules_]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._inputs(X)
        if len(X) == 0:
            return np.empty(0, dtype=np.float64)
        h = self.firing_strengths(X)
        local = np.c_[np.ones(len(X)), X] @ self.coef_.T
        pred = np.sum(h * local, axis=1)
        if self.prediction_clip is not None:
            pred = np.clip(pred, -self.prediction_clip, self.prediction_clip)
        if not np.isfinite(pred).all():
            raise FloatingPointError("non-finite DENFIS predictions")
        return pred

    def partial_fit(self, x: np.ndarray, y: float) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("partial_fit expects a single (d,) vector")
        self._inputs(x[None])
        if np.ndim(y) != 0 or not np.isfinite(y):
            raise ValueError("y must be a finite scalar")
        if self.evolve_rules:
            self._ecm_update(x)
            self._merge_rules()
        self._update_consequents(x, float(y))
        self.rule_history_.append(self.n_rules_)
