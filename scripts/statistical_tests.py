"""Statistical comparison of models from persisted predictions.

Run after the folds have been written:
    python scripts/statistical_tests.py

Produces:
* ``docs/statistical_tests.md`` — report-ready tables.
* ``results/statistical_tests.json`` — machine-readable results.

DM tests use squared-error loss and a Newey–West/Bartlett kernel.  A positive
DM statistic means model A has the larger squared errors (worse forecast).
Sharpe intervals use the stationary block bootstrap (block 20, 2000 draws,
seed from ``config.BOOTSTRAP_SEED``).  The 2026 holdout is excluded: it is a
single period reserved for the frozen final check.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import BOOTSTRAP_SEED, PREDS_DIR, RESULTS_DIR  # noqa: E402
from src.evaluation.backtest import sharpe  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.evaluation.stats import (  # noqa: E402
    bootstrap_sharpe_ci,
    dm_test,
    newey_west_lag,
)

FOLDS = [f"r{rep}f{fold}" for rep in (1, 2, 3) for fold in (1, 2, 3)]
PRED_COLUMNS = ["date", "y_true", "y_pred", "strat_ret"]


def discover_models() -> list[str]:
    """Model directories that contain every regular fold."""
    required = set(FOLDS)
    models = []
    for directory in sorted(PREDS_DIR.iterdir()):
        if not directory.is_dir():
            continue
        present = {path.stem for path in directory.glob("*.csv")}
        if required <= present:
            models.append(directory.name)
    if not models:
        raise SystemExit(f"No complete model predictions found under {PREDS_DIR}")
    return models


def load_pooled(model: str) -> pd.DataFrame:
    frames = [pd.read_csv(PREDS_DIR / model / f"{fold}.csv") for fold in FOLDS]
    pooled = pd.concat(frames, ignore_index=True)
    pooled["date"] = pd.to_datetime(pooled["date"])
    return pooled[PRED_COLUMNS].sort_values("date").reset_index(drop=True)


def align_predictions(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Inner-join two models on date and check that the targets agree."""
    merged = a.merge(b, on="date", how="inner", suffixes=("_a", "_b"))
    if merged.empty:
        raise ValueError("models have no test dates in common")
    if not np.allclose(merged["y_true_a"], merged["y_true_b"], atol=1e-10, rtol=0.0):
        raise ValueError("models disagree on y_true for shared dates")
    return merged.sort_values("date").reset_index(drop=True)


def pooled_row(model: str, frame: pd.DataFrame) -> dict:
    metrics = compute_metrics(frame["y_true"], frame["y_pred"])
    return {
        "model": model,
        "n": metrics["n"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "ic": metrics["ic"],
        "da": metrics["da"],
        "da5": metrics["da5"],
        "sharpe": sharpe(frame["strat_ret"].to_numpy(dtype=float)),
    }


def comparison_pairs(models: list[str], pooled: dict[str, dict]) -> list[tuple[str, str]]:
    """Primary DM comparisons: DENFIS vs the best baseline, plus frozen vs online."""
    denfis = sorted(m for m in models if m.startswith("denfis"))
    baselines = sorted(
        (m for m in models if not m.startswith("denfis")),
        key=lambda m: pooled[m]["rmse"],
    )

    pairs: list[tuple[str, str]] = []
    if denfis and baselines:
        reference = baselines[0]
        pairs.extend((model, reference) for model in denfis)
        if {"denfis_frozen", "denfis_online"} <= set(denfis):
            pairs.append(("denfis_online", "denfis_frozen"))
    elif len(baselines) >= 2:
        # Placeholder until DENFIS lands: best baseline vs runner-up.
        pairs.append((baselines[0], baselines[1]))
    return pairs


def _fmt(value) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    models = discover_models()
    pooled_frames = {model: load_pooled(model) for model in models}
    pooled = {model: pooled_row(model, pooled_frames[model]) for model in models}

    ci = {
        model: bootstrap_sharpe_ci(
            pooled_frames[model]["strat_ret"].to_numpy(dtype=float),
            n_boot=args.n_boot,
            avg_block=args.block,
            seed=args.seed,
        )
        for model in models
    }

    dm_rows = []
    for model_a, model_b in comparison_pairs(models, pooled):
        aligned = align_predictions(pooled_frames[model_a], pooled_frames[model_b])
        result = dm_test(
            aligned["y_true_a"],
            aligned["y_pred_a"],
            aligned["y_pred_b"],
        )
        outcome = "reject equal accuracy at 5%" if result["p_value"] < 0.05 else "not significant"
        dm_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "mean_loss_diff": result["mean_loss_diff"],
                "dm_stat": result["dm_stat"],
                "p_value": result["p_value"],
                "lag": result["lag"],
                "n": result["n"],
                "outcome": outcome,
                "a_worse": result["dm_stat"] > 0,
            }
        )

    has_denfis = any(model.startswith("denfis") for model in models)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_years": "2017-2025 (nine regular folds, holdout excluded)",
        "pooled": pooled,
        "sharpe_ci": ci,
        "dm": dm_rows,
        "denfis_available": has_denfis,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "statistical_tests.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    pooled_rows = [
        {
            "model": model,
            "n": pooled[model]["n"],
            "rmse": pooled[model]["rmse"],
            "mae": pooled[model]["mae"],
            "r2": pooled[model]["r2"],
            "ic": pooled[model]["ic"],
            "da": pooled[model]["da"],
            "sharpe": pooled[model]["sharpe"],
            "sharpe_95ci": f"[{ci[model]['lo']:.3f}, {ci[model]['hi']:.3f}]",
        }
        for model in models
    ]
    dm_table_rows = [
        {
            **row,
            "p_value": f"{row['p_value']:.4g}",
            "mean_loss_diff": f"{row['mean_loss_diff']:+.2e}",
        }
        for row in dm_rows
    ]

    lines = [
        "# Statistical tests",
        "",
        "Pooled out-of-sample results, 2017–2025 (nine walk-forward folds; "
        "the 2026 holdout is excluded).",
        "",
        "## Pooled performance and Sharpe confidence intervals",
        "",
        markdown_table(
            pooled_rows,
            ["model", "n", "rmse", "mae", "r2", "ic", "da", "sharpe", "sharpe_95ci"],
        ),
        "",
        f"Sharpe CI: stationary block bootstrap, average block {args.block} days, "
        f"{args.n_boot} resamples, seed {args.seed} (protocol section 6.5).",
        "",
        "## Diebold–Mariano test (squared-error loss, Newey–West)",
        "",
        f"Positive DM statistic = model A has the larger squared errors "
        f"(worse forecast). Lag uses the standard rule "
        f"`floor(4 * (n/100)^(2/9))`; n here is the aligned pooled sample.",
        "",
    ]
    if dm_table_rows:
        lines.append(
            markdown_table(
                dm_table_rows,
                [
                    "model_a",
                    "model_b",
                    "mean_loss_diff",
                    "dm_stat",
                    "p_value",
                    "lag",
                    "n",
                    "outcome",
                ],
            )
        )
    else:
        lines.append("_No comparable pairs available._")
    lines += [
        "",
        "## Reading the table",
        "",
        "- `mean_loss_diff` is mean squared error of A minus that of B; a positive "
        "value means A is worse.",
        "- A significant positive DM statistic means the better model is genuinely "
        "more accurate, not just luckier on these folds.",
        "- The Sharpe interval resamples contiguous blocks of days, preserving some "
        "autocorrelation; an interval spanning zero means the trading result is not "
        "distinguishable from no edge.",
    ]
    if not has_denfis:
        lines += [
            "",
            "> **Placeholder:** DENFIS predictions are not on disk yet, so the DM "
            "row compares the two best baselines. Re-run this script after the "
            "DENFIS folds land; the pair selection will switch automatically to "
            "DENFIS vs the best baseline.",
        ]

    docs_path = ROOT / "docs" / "statistical_tests.md"
    docs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {docs_path}")
    print(f"Wrote {json_path}")
    print()
    print(markdown_table(pooled_rows, ["model", "rmse", "da", "sharpe", "sharpe_95ci"]))
    if dm_table_rows:
        print()
        print(
            markdown_table(
                dm_table_rows,
                ["model_a", "model_b", "dm_stat", "p_value", "outcome"],
            )
        )


if __name__ == "__main__":
    main()
