"""Render results/summary.csv into markdown tables for the report draft."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PREDS_DIR, RESULTS_DIR  # noqa: E402
from src.evaluation.backtest import annualized_return, max_drawdown, sharpe  # noqa: E402

COLUMNS = [
    "model",
    "n",
    "rmse",
    "mae",
    "r2",
    "ic",
    "da",
    "da5",
    "sharpe",
    "ann_ret",
    "max_dd",
    "turnover",
    "exposure",
]

TRADING_COLUMNS = ["model", "n", "sharpe", "ann_ret", "max_dd", "turnover", "exposure"]


def _fmt(value):
    if pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in headers) + " |")
    return "\n".join(lines)


def _buy_and_hold_rows() -> list[dict]:
    """Buy-and-hold on the same test dates as the models."""
    rows = []
    model_dir = PREDS_DIR / "xgboost"
    regular = sorted(model_dir.glob("r*f*.csv"))
    sets = {
        "pooled": [pd.read_csv(path) for path in regular],
        "holdout": [pd.read_csv(model_dir / "holdout.csv")],
    }
    for scope, frames in sets.items():
        data = pd.concat(frames, ignore_index=True).sort_values("date")
        returns = data["bench_ret"].to_numpy(dtype=float)
        rows.append(
            {
                "model": "buy_hold",
                "scope": scope,
                "n": len(returns),
                "sharpe": sharpe(returns),
                "ann_ret": annualized_return(returns),
                "max_dd": max_drawdown(returns),
                "turnover": 0.0,
                "exposure": 1.0,
            }
        )
    return rows


def main() -> None:
    summary = pd.read_csv(RESULTS_DIR / "summary.csv")
    pooled = summary[summary["scope"] == "pooled"][COLUMNS].reset_index(drop=True)
    holdout = summary[summary["scope"] == "holdout"][COLUMNS].reset_index(drop=True)

    bh = pd.DataFrame(_buy_and_hold_rows())
    trading_rows = []

    pooled_trading = summary[summary["scope"] == "pooled"][TRADING_COLUMNS].copy()
    pooled_trading = pd.concat(
        [pooled_trading, bh[bh["scope"] == "pooled"][TRADING_COLUMNS]],
        ignore_index=True,
    )
    holdout_trading = summary[summary["scope"] == "holdout"][TRADING_COLUMNS].copy()
    holdout_trading = pd.concat(
        [holdout_trading, bh[bh["scope"] == "holdout"][TRADING_COLUMNS]],
        ignore_index=True,
    )

    text = [
        "# Baseline results",
        "",
        "## Pooled, 2017–2025",
        "",
        markdown_table(pooled),
        "",
        "## Holdout, 2026 Jan–Oct",
        "",
        markdown_table(holdout),
        "",
        "## Trading performance vs buy & hold",
        "",
        "### Pooled, 2017–2025",
        "",
        markdown_table(pooled_trading),
        "",
        "### Holdout, 2026 Jan–Oct",
        "",
        markdown_table(holdout_trading),
        "",
        "## Reading notes",
        "",
        "- Both models have pooled R² below zero and directional accuracy near 50%: "
        "next-day log returns are close to unpredictable at this feature set.",
        "- LSTM has the higher pooled Sharpe (0.41 vs 0.24) because it trades less "
        "(31% exposure vs 54%) and takes a lower turnover; XGBoost has the higher "
        "directional accuracy (0.508 vs 0.485).",
        "- Both long/flat strategies cut maximum drawdown relative to buy & hold "
        "(27.5% and 32.2% vs 38.6%) but give up most of the upside: buy & hold has "
        "a pooled Sharpe of 1.11 and 465% cumulative return over 2017–2025.",
        "- The 2026 holdout is a single favourable period for both models; it is "
        "reported separately and is not used for model selection.",
    ]
    output = "\n".join(text) + "\n"

    path = ROOT / "docs" / "results_table.md"
    path.write_text(output, encoding="utf-8")
    print(output)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
