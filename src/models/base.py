"""Minimal model protocols shared by every model in the project.

A model receives a float64 matrix and returns predictions in log-return
units.  The CV runner only relies on ``fit`` and ``predict``; models that can
use an explicit validation set may additionally implement
``fit_with_validation``, and online models may implement ``partial_fit``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Predictor(Protocol):
    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


@runtime_checkable
class OnlinePredictor(Predictor, Protocol):
    def partial_fit(self, x: np.ndarray, y: float) -> None:
        ...


@runtime_checkable
class ValidationAware(Protocol):
    def fit_with_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        ...
