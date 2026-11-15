"""B2: equal-weight ensemble of three single-rule DENFIS components.

Each component is a one-rule DENFIS model on one feature plus the two
ex-dividend dummies (``is_exdiv``, ``is_post_exdiv_1``).  The ensemble passes
the full input matrix to every component and lets each select its own columns,
so the CV runner can fit one scaler on the union of the columns inside the same
training window.

The class implements the interface the CV runner expects from a tabular model:
``fit``, ``predict``, ``get_params`` and ``n_rules_``.  It never touches dates,
prices or files.
"""

from __future__ import annotations

import numpy as np

from config import GLOBAL_SEED
from src.models.denfis import DENFIS


class DENFISEnsemble:
    """Equal-weight average of one-rule DENFIS models on column subsets."""

    name = "denfis_b2"

    def __init__(
        self,
        components: list[list[int]],
        dthr: float = 3.0,
        lam: float = 1.0,
        max_rules: int = 1,
        seed: int = GLOBAL_SEED,
        prediction_clip: float | None = 0.2,
    ):
        if not components:
            raise ValueError("components must not be empty")
        self.components = [list(map(int, comp)) for comp in components]
        if any(len(comp) == 0 for comp in self.components):
            raise ValueError("each component needs at least one column")
        self.dthr, self.lam = float(dthr), float(lam)
        self.max_rules = int(max_rules)
        self.seed = int(seed)
        self.prediction_clip = prediction_clip
        self.models: list[DENFIS] = []

    @property
    def n_rules_(self) -> int:
        return int(sum(model.n_rules_ for model in self.models)) if self.models else 0

    def get_params(self) -> dict:
        return {
            "dthr": self.dthr,
            "lam": self.lam,
            "max_rules": self.max_rules,
            "seed": self.seed,
            "prediction_clip": self.prediction_clip,
            "components": [list(comp) for comp in self.components],
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be (n, d) and match y")
        self.models = []
        for index, comp in enumerate(self.components):
            model = DENFIS(
                dthr=self.dthr,
                lam=self.lam,
                max_rules=self.max_rules,
                seed=self.seed + index,
                prediction_clip=self.prediction_clip,
            )
            model.fit(X[:, comp], y)
            self.models.append(model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("fit must be called before predict")
        X = np.asarray(X, dtype=np.float64)
        parts = [model.predict(X[:, comp]) for model, comp in zip(self.models, self.components)]
        return np.mean(parts, axis=0)


__all__ = ["DENFISEnsemble"]
