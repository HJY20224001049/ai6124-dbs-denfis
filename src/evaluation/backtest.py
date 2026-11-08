"""Long/flat backtest on simple returns, with costs and threshold selection.

Predictions and ``y_true`` live in log-return units.  The backtest converts
realised log returns to simple returns before compounding, so the equity curve
is financially correct.  Cost is charged on absolute position changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import ANNUALIZATION, COST_BPS, TAU_GRID


def log_to_simple(log_returns) -> np.ndarray:
    return np.expm1(np.asarray(log_returns, dtype=np.float64).ravel())


def make_signal(y_pred, tau: float) -> np.ndarray:
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return (y_pred > tau).astype(np.int64)


def run_backtest(
    signal,
    simple_returns,
    cost_bps: float = COST_BPS,
) -> pd.DataFrame:
    """Return per-day signal, turnover, strategy return, and benchmark return."""
    signal = np.asarray(signal, dtype=np.int64).ravel()
    simple_returns = np.asarray(simple_returns, dtype=np.float64).ravel()
    if signal.shape != simple_returns.shape:
        raise ValueError(
            f"shape mismatch: signal {signal.shape} vs returns {simple_returns.shape}"
        )

    prev = np.concatenate([[0], signal[:-1]])
    turnover = np.abs(signal - prev).astype(np.float64)
    cost = (cost_bps / 10_000.0) * turnover
    strat_ret = signal * simple_returns - cost

    return pd.DataFrame(
        {
            "signal": signal,
            "position": signal,
            "turnover": turnover,
            "strat_ret": strat_ret,
            "bench_ret": simple_returns,
        }
    )


def equity_curve(returns) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float64).ravel()
    return np.concatenate([[1.0], np.cumprod(1.0 + returns)])


def max_drawdown(returns) -> float:
    """Peak-to-trough drawdown as a positive fraction (0.20 = 20%)."""
    equity = equity_curve(returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(-drawdown.min())


def sharpe(returns, annualization: int = ANNUALIZATION) -> float:
    returns = np.asarray(returns, dtype=np.float64).ravel()
    if returns.size < 2 or returns.std(ddof=1) == 0.0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(annualization))


def annualized_return(returns, annualization: int = ANNUALIZATION) -> float:
    returns = np.asarray(returns, dtype=np.float64).ravel()
    if returns.size == 0:
        return 0.0
    growth = float(np.prod(1.0 + returns))
    if growth <= 0.0:
        return -1.0
    return float(growth ** (annualization / returns.size) - 1.0)


def turnover(signal) -> float:
    signal = np.asarray(signal, dtype=np.int64).ravel()
    if signal.size == 0:
        return 0.0
    prev = np.concatenate([[0], signal[:-1]])
    return float(np.mean(np.abs(signal - prev)))


def exposure(signal) -> float:
    signal = np.asarray(signal, dtype=np.int64).ravel()
    return float(np.mean(signal)) if signal.size else 0.0


def backtest_metrics(strat_ret, signal) -> dict[str, float]:
    strat_ret = np.asarray(strat_ret, dtype=np.float64).ravel()
    return {
        "n": int(strat_ret.size),
        "sharpe": sharpe(strat_ret),
        "ann_ret": annualized_return(strat_ret),
        "max_dd": max_drawdown(strat_ret),
        "turnover": turnover(signal),
        "exposure": exposure(signal),
        "cum_return": float(np.prod(1.0 + strat_ret) - 1.0),
    }


def select_tau(
    y_true_log,
    y_pred_log,
    tau_grid=TAU_GRID,
    cost_bps: float = COST_BPS,
) -> float:
    """Pick the validation threshold with the highest Sharpe; floor at zero.

    Ties go to the smaller threshold.  If every candidate has a negative
    Sharpe, the protocol's fallback is ``tau = 0``.
    """
    simple = log_to_simple(y_true_log)
    best_tau = 0.0
    best_sharpe = -np.inf
    for tau in sorted(tau_grid):
        signal = make_signal(y_pred_log, tau)
        bt = run_backtest(signal, simple, cost_bps=cost_bps)
        value = sharpe(bt["strat_ret"].to_numpy())
        if value > best_sharpe:
            best_sharpe = value
            best_tau = float(tau)
    if best_sharpe < 0.0:
        return 0.0
    return best_tau
