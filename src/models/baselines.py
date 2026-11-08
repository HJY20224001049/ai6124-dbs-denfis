"""Benchmark models.

XGBoost is implemented first because it validates the CV runner end to end.
The LSTM is added to this module once the sequence plumbing is in place.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from xgboost import XGBRegressor

from config import GLOBAL_SEED


class XGBModel:
    """XGBoost regressor with the protocol's fixed search space."""

    name = "xgboost"

    def __init__(self, params: Optional[dict] = None, seed: int = GLOBAL_SEED):
        self.params = dict(params or {})
        self.seed = seed
        self.model: Optional[XGBRegressor] = None
        self.best_iteration_: Optional[int] = None

    def _build(self, n_estimators: int, early_stopping: bool) -> XGBRegressor:
        kwargs = dict(
            n_estimators=int(n_estimators),
            max_depth=int(self.params.get("max_depth", 2)),
            learning_rate=float(self.params.get("learning_rate", 0.03)),
            subsample=float(self.params.get("subsample", 0.8)),
            colsample_bytree=float(self.params.get("colsample_bytree", 0.8)),
            min_child_weight=float(self.params.get("min_child_weight", 5)),
            reg_lambda=float(self.params.get("reg_lambda", 1.0)),
            objective="reg:squarederror",
            tree_method="hist",
            random_state=self.seed,
            n_jobs=1,
        )
        if early_stopping:
            kwargs["early_stopping_rounds"] = 50
        return XGBRegressor(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_estimators = int(self.params.get("n_estimators", 500))
        if self.best_iteration_ is not None:
            n_estimators = self.best_iteration_ + 1
        self.model = self._build(n_estimators, early_stopping=False)
        self.model.fit(X, y, verbose=False)

    def fit_with_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        self.model = self._build(1000, early_stopping=True)
        self.model.fit(
            X,
            y,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best = getattr(self.model, "best_iteration", None)
        self.best_iteration_ = int(best) if best is not None else None

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("XGBModel.predict called before fit")
        return np.asarray(self.model.predict(X), dtype=np.float64)


class _LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        return self.head(self.dropout(last)).squeeze(-1)


class LSTMModel:
    """Small CPU LSTM over 20-day sequences of the same 13 features.

    ``fit`` expects a 3-D array ``(n, seq_len, n_features)``.  The CV runner
    builds those sequences with context from earlier rows, so a test sequence
    never sees future prices.
    """

    name = "lstm"
    requires_sequence = True

    def __init__(self, params: Optional[dict] = None, seed: int = GLOBAL_SEED):
        self.params = dict(params or {})
        self.seed = seed
        self.seq_len = int(self.params.get("seq_len", 20))
        self.hidden_size = int(self.params.get("hidden_size", 32))
        self.dropout = float(self.params.get("dropout", 0.1))
        self.epochs = int(self.params.get("epochs", 100))
        self.patience = int(self.params.get("patience", 15))
        self.batch_size = int(self.params.get("batch_size", 64))
        self.lr = float(self.params.get("lr", 1e-3))
        self.weight_decay = float(self.params.get("weight_decay", 1e-4))
        self.model: Optional[_LSTMRegressor] = None
        self.best_epoch_: Optional[int] = None

    def _score(self, X: torch.Tensor, y: np.ndarray) -> float:
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X).cpu().numpy()
        return float(np.sqrt(np.mean((y - pred) ** 2)))

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        max_epochs: Optional[int] = None,
    ) -> None:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"LSTM expects (n, seq_len, features), got {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")
        if len(X) == 0:
            raise ValueError("empty training set")

        max_epochs = self.epochs if max_epochs is None else int(max_epochs)
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older torch
            torch.use_deterministic_algorithms(True)

        self.model = _LSTMRegressor(X.shape[2], self.hidden_size, self.dropout)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        use_validation = X_val is not None and y_val is not None
        if use_validation:
            X_val_t = torch.from_numpy(np.asarray(X_val, dtype=np.float32))
            y_val_np = np.asarray(y_val, dtype=np.float32)

        best_state = None
        best_val = np.inf
        best_epoch = -1
        wait = 0

        for epoch in range(max_epochs):
            self.model.train()
            generator = torch.Generator().manual_seed(self.seed + epoch)
            perm = torch.randperm(len(X_t), generator=generator)
            for start in range(0, len(X_t), self.batch_size):
                batch_idx = perm[start : start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(X_t[batch_idx])
                loss = loss_fn(pred, y_t[batch_idx])
                loss.backward()
                optimizer.step()

            if use_validation:
                val_rmse = self._score(X_val_t, y_val_np)
                if val_rmse < best_val - 1e-12:
                    best_val = val_rmse
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

        if use_validation and best_state is not None:
            self.model.load_state_dict(best_state)
            self.best_epoch_ = int(best_epoch) + 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        epochs = self.best_epoch_ if self.best_epoch_ is not None else self.epochs
        self._fit(X, y, max_epochs=epochs)

    def fit_with_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        self._fit(X, y, X_val=X_val, y_val=y_val, max_epochs=self.epochs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LSTMModel.predict called before fit")
        X = np.asarray(X, dtype=np.float32)
        self.model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(X), 256):
                chunks.append(self.model(torch.from_numpy(X[start : start + 256])).cpu().numpy())
        return np.concatenate(chunks).astype(np.float64)
