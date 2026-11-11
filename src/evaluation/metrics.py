"""Regression and directional metrics on log-return predictions."""

from __future__ import annotations

import numpy as np

from config import DEADBAND


def _pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        raise ValueError("no finite prediction pairs")
    return y_true, y_pred


def rmse(y_true, y_pred) -> float:
    yt, yp = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    yt, yp = _pair(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def r2(y_true, y_pred) -> float:
    yt, yp = _pair(y_true, y_pred)
    sst = float(np.sum((yt - yt.mean()) ** 2))
    if sst == 0.0:
        return 0.0
    sse = float(np.sum((yt - yp) ** 2))
    return float(1.0 - sse / sst)


def information_coefficient(y_true, y_pred) -> float:
    """Pearson correlation between prediction and realised return."""
    yt, yp = _pair(y_true, y_pred)
    if yt.std(ddof=0) == 0.0 or yp.std(ddof=0) == 0.0:
        return 0.0
    return float(np.corrcoef(yt, yp)[0, 1])


def directional_accuracy(y_true, y_pred, deadband: float = 0.0) -> float:
    """Share of days whose predicted and realised signs agree.

    ``sign(0) == 0``.  With ``deadband > 0``, days with ``|y_true|`` below the
    deadband are excluded (the 5 bp version reported in the protocol).
    """
    yt, yp = _pair(y_true, y_pred)
    if deadband > 0:
        keep = np.abs(yt) >= deadband
        yt, yp = yt[keep], yp[keep]
        if yt.size == 0:
            return float("nan")
    return float(np.mean(np.sign(yt) == np.sign(yp)))


def zero_baseline_metrics(y_true) -> dict[str, float]:
    """Metrics of the no-prediction forecast ``y_pred = 0``.

    In return space this is the random-walk forecast: the price is assumed to
    stay at its last observed value.  Its RMSE is ``sqrt(mean(y**2))`` and its
    MAE is ``mean(|y|)``; every model's skill is measured against these.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yt = yt[np.isfinite(yt)]
    if yt.size == 0:
        raise ValueError("no finite targets")
    return {
        "rmse_zero": float(np.sqrt(np.mean(yt**2))),
        "mae_zero": float(np.mean(np.abs(yt))),
    }


def rmse_skill(y_true, y_pred) -> float:
    """Relative RMSE improvement over the no-prediction forecast.

    ``skill = 1 - RMSE(model) / RMSE(zeros)``.  Positive means the model
    beats "predict no change"; negative means it is worse.
    """
    yt, yp = _pair(y_true, y_pred)
    baseline = float(np.sqrt(np.mean(yt**2)))
    if baseline == 0.0:
        return float("nan")
    return float(1.0 - np.sqrt(np.mean((yt - yp) ** 2)) / baseline)


def mae_skill(y_true, y_pred) -> float:
    """Relative MAE improvement over the no-prediction forecast."""
    yt, yp = _pair(y_true, y_pred)
    baseline = float(np.mean(np.abs(yt)))
    if baseline == 0.0:
        return float("nan")
    return float(1.0 - np.mean(np.abs(yt - yp)) / baseline)


def r2_zero(y_true, y_pred) -> float:
    """Coefficient of determination against the no-prediction forecast.

    ``1 - SSE(model) / SSE(zeros)``; zero means the model ties the random
    walk, negative means it is worse.
    """
    yt, yp = _pair(y_true, y_pred)
    sse_zero = float(np.sum(yt**2))
    if sse_zero == 0.0:
        return float("nan")
    return float(1.0 - np.sum((yt - yp) ** 2) / sse_zero)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    yt, yp = _pair(y_true, y_pred)
    baseline = zero_baseline_metrics(yt)
    metrics = {
        "n": int(yt.size),
        "rmse": rmse(yt, yp),
        "mae": mae(yt, yp),
        "r2": r2(yt, yp),
        "ic": information_coefficient(yt, yp),
        "da": directional_accuracy(yt, yp),
        "da5": directional_accuracy(yt, yp, deadband=DEADBAND),
        "rmse_zero": baseline["rmse_zero"],
        "mae_zero": baseline["mae_zero"],
        "rmse_skill": rmse_skill(yt, yp),
        "mae_skill": mae_skill(yt, yp),
        "r2_zero": r2_zero(yt, yp),
    }
    return metrics
