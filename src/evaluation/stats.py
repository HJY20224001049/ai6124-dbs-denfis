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


def _newey_west_mean_variance(values: np.ndarray, lag: int) -> float:
    """Variance of the sample mean with a Bartlett/Newey–West kernel."""
    values = np.asarray(values, dtype=np.float64)
    demeaned = values - values.mean()
    long_run = float(np.mean(demeaned**2))
    for ell in range(1, lag + 1):
        weight = 1.0 - ell / (lag + 1.0)
        long_run += 2.0 * weight * float(np.mean(demeaned[ell:] * demeaned[:-ell]))
    return long_run / len(values)


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

    long_run = _newey_west_mean_variance(d, lag) * n

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


def clark_west(
    y_true,
    pred_small,
    pred_large,
    *,
    lag: int | None = None,
) -> dict:
    """Clark–West (2007) test for nested forecast models.

    ``pred_small`` is the restricted model (typically the zero forecast) and
    ``pred_large`` the model that nests it.  The adjusted loss differential is

    ``d_t = e_small,t² − e_large,t² + (pred_small,t − pred_large,t)²``

    and the null is that the large model's extra parameters are zero.  The
    statistic is one-sided: a positive value rejects that null, i.e. the
    forecast carries information.  It is *not* an MSE-improvement test: the
    adjustment credits the forecast's own variance, so a positive statistic can
    coexist with a worse raw RMSE when forecasts are over-dispersed.  The
    adjustment is what makes the statistic standard normal under the null,
    which plain Diebold–Mariano does not deliver for nested models.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    ps = np.asarray(pred_small, dtype=np.float64).ravel()
    pl = np.asarray(pred_large, dtype=np.float64).ravel()
    if not (yt.shape == ps.shape == pl.shape):
        raise ValueError(f"shape mismatch: y {yt.shape}, small {ps.shape}, large {pl.shape}")
    mask = np.isfinite(yt) & np.isfinite(ps) & np.isfinite(pl)
    yt, ps, pl = yt[mask], ps[mask], pl[mask]
    n = yt.size
    if n < 3:
        raise ValueError("need at least three aligned observations")

    d = (yt - ps) ** 2 - (yt - pl) ** 2 + (ps - pl) ** 2
    mean_d = float(np.mean(d))
    if lag is None:
        lag = newey_west_lag(n)
    lag = int(min(lag, n - 1))
    variance = _newey_west_mean_variance(d, lag)

    if mean_d == 0.0:
        stat, p_value = 0.0, 0.5
    elif variance <= 0.0:
        stat = float("inf") if mean_d > 0 else 0.0
        p_value = 0.0 if mean_d > 0 else 1.0
    else:
        stat = mean_d / np.sqrt(variance)
        p_value = float(scipy_stats.norm.sf(stat))

    return {
        "cw_stat": float(stat),
        "p_value": p_value,
        "mean_loss_diff": mean_d,
        "lag": lag,
        "n": int(n),
        "large_better": bool(stat > 0),
    }


def model_confidence_set(
    losses,
    names: list[str] | None = None,
    *,
    alpha: float = 0.10,
    n_boot: int = DEFAULT_N_BOOT,
    avg_block: int = DEFAULT_BLOCK,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Range-based model confidence set (Hansen, Lunde & Nason 2011).

    ``losses`` is an ``(n, m)`` matrix of per-observation losses on the same
    dates.  The routine repeatedly eliminates the model with the largest
    standardised excess loss until the range statistic falls below its
    bootstrap critical value.  The surviving set contains every model that
    cannot be separated from the best one at level ``alpha``.

    The bootstrap is the stationary bootstrap of Politis & Romano (1994); the
    same resampled means are reused across elimination steps, which is the
    standard implementation shortcut for a small model set.
    """
    L = np.asarray(losses, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] < 3 or L.shape[1] < 2:
        raise ValueError("losses must be (n, m) with n >= 3 and m >= 2")
    if not np.isfinite(L).all():
        raise ValueError("losses must be finite")
    m = L.shape[1]
    labels = list(names) if names is not None else [f"model_{i}" for i in range(m)]
    if len(labels) != m:
        raise ValueError("names must match the number of loss columns")

    indices = [
        stationary_bootstrap_indices(L.shape[0], avg_block, seed=seed + b)
        for b in range(n_boot)
    ]
    boot_means = np.array([L[idx].mean(axis=0) for idx in indices])

    remaining = list(range(m))
    eliminated: list[dict] = []
    while len(remaining) > 1:
        block = L[:, remaining]
        mean_loss = block.mean(axis=0)
        boot = boot_means[:, remaining]
        diff_boot = boot[:, :, None] - boot[:, None, :]          # (B, k, k)
        diff_mean = mean_loss[:, None] - mean_loss[None, :]      # (k, k)
        variance = diff_boot.var(axis=0, ddof=1)
        safe = np.maximum(variance, 1e-300)
        t = diff_mean / np.sqrt(safe)
        np.fill_diagonal(t, -np.inf)
        t_star = (diff_boot - diff_mean[None]) / np.sqrt(safe)[None]
        for b in range(t_star.shape[0]):
            np.fill_diagonal(t_star[b], 0.0)
        tr = float(np.max(t))
        tr_star = np.max(np.abs(t_star), axis=(1, 2))
        critical = float(np.quantile(tr_star, 1.0 - alpha))
        if tr <= critical:
            break
        worst = int(np.argmax(np.max(t, axis=1)))
        eliminated.append({
            "model": labels[remaining[worst]],
            "tr": tr,
            "critical": critical,
            "step": len(eliminated) + 1,
        })
        remaining.pop(worst)

    return {
        "included": [labels[i] for i in remaining],
        "eliminated": eliminated,
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "avg_block": int(avg_block),
        "seed": int(seed),
    }
