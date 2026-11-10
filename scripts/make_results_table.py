"""Render results/summary.csv into markdown tables for the report draft."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The generated markdown contains characters (R², en dashes) that the default
# Windows console encoding cannot print.  The file itself is always UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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


FOLDS = [f"r{rep}f{fold}" for rep in (1, 2, 3) for fold in (1, 2, 3)]


def _reference_model() -> str:
    """Any model with all nine folds plus the holdout; prefer the main model."""
    required = set(FOLDS) | {"holdout"}
    complete = []
    for directory in sorted(PREDS_DIR.iterdir()):
        if not directory.is_dir():
            continue
        present = {path.stem for path in directory.glob("*.csv")}
        if required <= present:
            complete.append(directory.name)
    if not complete:
        raise SystemExit("No model has all nine folds and the holdout on disk.")
    for preferred in ("denfis_frozen", "xgboost", "lstm"):
        if preferred in complete:
            return preferred
    return complete[0]


def _buy_and_hold_rows() -> list[dict]:
    """Buy-and-hold on the same test dates as the models."""
    rows = []
    model_dir = PREDS_DIR / _reference_model()
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


def _reading_notes(summary: pd.DataFrame, buy_hold: pd.DataFrame) -> list[str]:
    """Generate interpretation notes from the current result table."""
    pooled = summary[summary["scope"] == "pooled"].set_index("model")
    notes: list[str] = []
    if pooled.empty:
        return notes

    r2_text = ", ".join(f"{model} {pooled.loc[model, 'r2']:.4f}" for model in pooled.index)
    positive = [model for model in pooled.index if pooled.loc[model, "r2"] > 0]
    if positive:
        notes.append(
            f"Pooled R² is positive for {', '.join(positive)} and negative for the "
            f"others ({r2_text}); the test years are not uniformly favourable."
        )
    else:
        notes.append(
            f"Pooled R² is below zero for every model ({r2_text}): next-day log "
            "returns are close to unpredictable with this feature set."
        )

    sharpe_text = ", ".join(
        f"{model} {pooled.loc[model, 'sharpe']:.2f} "
        f"(exposure {pooled.loc[model, 'exposure']:.0%})"
        for model in pooled.sort_values("sharpe", ascending=False).index
    )
    notes.append(f"Pooled Sharpe ranking: {sharpe_text}.")

    da_text = ", ".join(
        f"{model} {pooled.loc[model, 'da']:.3f}" for model in pooled.index
    )
    notes.append(f"Directional accuracy: {da_text} (0.5 is a coin flip).")

    bh_pooled = buy_hold[buy_hold["scope"] == "pooled"]
    if not bh_pooled.empty:
        bh_dd = float(bh_pooled["max_dd"].iloc[0])
        dd_text = ", ".join(
            f"{model} {pooled.loc[model, 'max_dd']:.1%}" for model in pooled.index
        )
        notes.append(
            f"Pooled maximum drawdown: {dd_text}, versus {bh_dd:.1%} for buy & hold. "
            "Lower drawdown only counts as an improvement if the Sharpe is not "
            "sacrificed."
        )

    notes.append(
        "The 2026 holdout is a single period reported separately; it is not used "
        "for model selection and should not be treated as a second validation set."
    )
    return notes


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

    has_denfis = (
        summary["model"].astype(str).str.startswith("denfis").any()
        if not summary.empty
        else False
    )
    title = "# Results" if has_denfis else "# Baseline results"
    text = [
        title,
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
        *[f"- {note}" for note in _reading_notes(summary, bh)],
    ]
    output = "\n".join(text) + "\n"

    path = ROOT / "docs" / "results_table.md"
    path.write_text(output, encoding="utf-8")
    print(output)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
