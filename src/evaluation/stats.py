"""Statistical tests for the model comparison (protocol section 6.5).

Two tools are provided:

* ``dm_test`` — Diebold–Mariano test with squared-error loss and a
  Newey–West/Bartlett kernel, lag chosen by the standard rule of thumb.
* ``bootstrap_sharpe_ci`` — 95% confidence interval for the annualised
  Sharpe ratio using the stationary block bootstrap of Politis & Romano
  (1994), average block length 20 days and 2000 resamples by default.

Both functions operate on aligned, daily observations and never look at
future data.  The DM sign convention is: a positive statistic means
``pred_a`` has the larger squared errors (i.e. the worse forecast).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

from config import ANNUALIZATION, BOOTSTRAP_SEED
from src.evaluation.backtest import sharpe

DEFAULT_BLOCK = 20
DEFAULT_N_BOOT = 2000


def newey_west_lag(n: int) -> int:
    """Standard Newey–West (1994) plug-in lag: floor(4 * (n/100)^(2/9))."""
    if n <= 0:
        raise ValueError("n must be positive")
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _squared_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)) ** 2


def dm_test(
    y_true,
    pred_a,
    pred_b,
    *,
    lag: int | None = None,
    loss: str = "squared",
) -> dict:
    """Diebold–Mariano test of equal predictive accuracy.

    Parameters
    ----------
    y_true, pred_a, pred_b : array-like
        Realised target and the two sets of predictions, on the same dates.
    lag : int, optional
        Newey–West lag; defaults to :func:`newey_west_lag`.
    loss : {"squared", "absolute"}
        Loss function.  The protocol freezes squared-error loss.

    Returns
    -------
    dict with ``dm_stat`` (positive when ``pred_a`` is worse), two-sided
    ``p_value`` from the standard normal approximation, the mean loss
    differential, the lag and the number of observations.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    pa = np.asarray(pred_a, dtype=np.float64).ravel()
    pb = np.asarray(pred_b, dtype=np.float64).ravel()
    if not (yt.shape == pa.shape == pb.shape):
        raise ValueError(
            f"shape mismatch: y {yt.shape}, a {pa.shape}, b {pb.shape}"
        )
    mask = np.isfinite(yt) & np.isfinite(pa) & np.isfinite(pb)
    yt, pa, pb = yt[mask], pa[mask], pb[mask]
    n = yt.size
    if n < 2:
        raise ValueError("need at least two aligned observations")

    if loss == "squared":
        d = _squared_errors(yt, pa) - _squared_errors(yt, pb)
    elif loss == "absolute":
        d = np.abs(yt - pa) - np.abs(yt - pb)
    else:
        raise ValueError(f"unknown loss: {loss!r}")

    mean_d = float(np.mean(d))
    if lag is None:
        lag = newey_west_lag(n)
    lag = int(min(lag, n - 1))

    demeaned = d - mean_d
    gamma0 = float(np.mean(demeaned**2))
    long_run = gamma0
    for ell in range(1, lag + 1):
        weight = 1.0 - ell / (lag + 1.0)
        gamma = float(np.mean(demeaned[ell:] * demeaned[:-ell]))
        long_run += 2.0 * weight * gamma

    if mean_d == 0.0:
        dm_stat, p_value = 0.0, 1.0
    elif long_run <= 0.0:
        dm_stat = float(np.inf) if mean_d > 0 else float("-inf")
        p_value = 0.0
    else:
        dm_stat = mean_d / np.sqrt(long_run / n)
        p_value = float(2.0 * scipy_stats.norm.sf(abs(dm_stat)))

    return {
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "mean_loss_diff": mean_d,
        "lag": lag,
        "n": int(n),
        "loss": loss,
    }


def stationary_bootstrap_indices(
    n: int,
    avg_block: int = DEFAULT_BLOCK,
    seed: int | None = None,
) -> np.ndarray:
    """Resample positions with the stationary bootstrap (circular blocks).

    Each block starts with probability ``1 / avg_block``; block lengths are
    therefore geometric with mean ``avg_block`` and indices wrap around the
    sample.  Returns an integer array of length ``n``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if avg_block < 1:
        raise ValueError("avg_block must be at least 1")
    rng = np.random.default_rng(seed)
    p = 1.0 / float(avg_block)

    new_block = rng.random(n) < p
    new_block[0] = True
    base = rng.integers(0, n, size=n)

    run_id = np.cumsum(new_block) - 1
    run_starts = np.flatnonzero(new_block)
    base_at_start = base[run_starts]
    offsets = np.arange(n) - run_starts[run_id]
    return (base_at_start[run_id] + offsets) % n


def bootstrap_sharpe_ci(
    returns,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    avg_block: int = DEFAULT_BLOCK,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
    annualization: int = ANNUALIZATION,
) -> dict:
    """Percentile CI for the annualised Sharpe ratio of daily returns."""
    values = np.asarray(returns, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("need at least two finite returns")

    point = sharpe(values, annualization=annualization)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(values.size, avg_block, seed=seed + b)
        draws[b] = sharpe(values[idx], annualization=annualization)

    lo, hi = np.percentile(draws, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return {
        "point": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": int(n_boot),
        "avg_block": int(avg_block),
        "seed": int(seed),
        "alpha": float(alpha),
        "n": int(values.size),
    }
