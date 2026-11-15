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
    clark_west,
    dm_test,
    model_confidence_set,
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


def align_all(frames: dict[str, pd.DataFrame], models: list[str]) -> pd.DataFrame:
    """Inner-join every model's predictions on date into one loss matrix input."""
    merged = frames[models[0]][["date", "y_true", "y_pred"]].rename(
        columns={"y_pred": models[0]}
    )
    for model in models[1:]:
        part = frames[model][["date", "y_pred"]].rename(columns={"y_pred": model})
        merged = merged.merge(part, on="date", how="inner", validate="one_to_one")
    for model in models[1:]:
        other = frames[model].set_index("date")["y_true"].reindex(merged["date"])
        if not np.allclose(merged["y_true"], other, atol=1e-10, rtol=0.0):
            raise ValueError("models disagree on y_true for shared dates")
    return merged.sort_values("date").reset_index(drop=True)


def pooled_row(model: str, frame: pd.DataFrame) -> dict:
    metrics = compute_metrics(frame["y_true"], frame["y_pred"])
    return {
        "model": model,
        "n": metrics["n"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "rmse_zero": metrics["rmse_zero"],
        "rmse_skill": metrics["rmse_skill"],
        "r2": metrics["r2"],
        "ic": metrics["ic"],
        "da": metrics["da"],
        "da5": metrics["da5"],
        "sharpe": sharpe(frame["strat_ret"].to_numpy(dtype=float)),
    }


def comparison_pairs(models: list[str], pooled: dict[str, dict]) -> list[tuple[str, str]]:
    """DM comparisons against no prediction, the best baseline, and each other.

    Every predictive model is compared with ``no_prediction`` first.  DENFIS is
    then compared with the best *predictive* baseline (no prediction excluded,
    because it is not a model), and the frozen and online variants against each
    other.
    """
    no_prediction = "no_prediction"
    predictive = [model for model in models if model != no_prediction]
    denfis = sorted(m for m in predictive if m.startswith("denfis"))
    baselines = sorted(
        (m for m in predictive if not m.startswith("denfis")),
        key=lambda m: pooled[m]["rmse"],
    )

    pairs: list[tuple[str, str]] = []
    if no_prediction in models:
        pairs.extend((model, no_prediction) for model in denfis + baselines)
    if denfis and baselines:
        reference = baselines[0]
        pairs.extend((model, reference) for model in denfis)
    if {"denfis_frozen", "denfis_online"} <= set(denfis):
        pairs.append(("denfis_online", "denfis_frozen"))
    if not pairs and len(baselines) >= 2:
        # Placeholder until DENFIS lands: best baseline vs runner-up.
        pairs.append((baselines[0], baselines[1]))

    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            unique.append(pair)
    return unique


def _fmt(value, column: str = "") -> str:
    if column.endswith("_skill") and isinstance(value, (float, np.floating)):
        return f"{float(value):+.2%}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row[c], c) for c in columns) + " |")
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

    cw_rows = []
    if "no_prediction" in models:
        for model in models:
            if model == "no_prediction":
                continue
            aligned = align_predictions(pooled_frames[model], pooled_frames["no_prediction"])
            result = clark_west(aligned["y_true_a"], aligned["y_pred_b"], aligned["y_pred_a"])
            if result["p_value"] < 0.05 and result["cw_stat"] > 0:
                outcome = "rejects no-information null at 5%"
            elif result["cw_stat"] > 0:
                outcome = "positive, not significant"
            else:
                outcome = "no evidence of information"
            cw_rows.append(
                {
                    "model": model,
                    "mean_loss_diff": result["mean_loss_diff"],
                    "cw_stat": result["cw_stat"],
                    "p_value": result["p_value"],
                    "n": result["n"],
                    "outcome": outcome,
                }
            )

    common = align_all(pooled_frames, models)
    loss_matrix = np.column_stack(
        [(common["y_true"].to_numpy() - common[model].to_numpy()) ** 2 for model in models]
    )
    mcs = model_confidence_set(
        loss_matrix, models, alpha=0.10,
        n_boot=args.n_boot, avg_block=args.block, seed=args.seed,
    )

    mde_rows = []
    mde_path = RESULTS_DIR / "analysis" / "detectable_skill.csv"
    if mde_path.exists():
        mde = pd.read_csv(mde_path)
        for _, row in mde.iterrows():
            mde_rows.append(
                {
                    "comparison": str(row["rule"]),
                    "observed_rmse_skill": float(row["observed_rmse_skill"]),
                    "mde_rmse_skill": float(row["mde_rmse_skill"]),
                    "power_at_observed": float(row["power_at_observed"]),
                    "n": int(row["n"]),
                }
            )

    has_denfis = any(model.startswith("denfis") for model in models)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_years": "2017-2025 (nine regular folds, holdout excluded)",
        "pooled": pooled,
        "sharpe_ci": ci,
        "dm": dm_rows,
        "clark_west": cw_rows,
        "model_confidence_set": mcs,
        "mde": mde_rows,
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
            "rmse_zero": pooled[model]["rmse_zero"],
            "rmse_skill": pooled[model]["rmse_skill"],
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
            [
                "model",
                "n",
                "rmse",
                "rmse_zero",
                "rmse_skill",
                "mae",
                "r2",
                "ic",
                "da",
                "sharpe",
                "sharpe_95ci",
            ],
        ),
        "",
        f"Sharpe CI: stationary block bootstrap, average block {args.block} days, "
        f"{args.n_boot} resamples, seed {args.seed} (protocol section 6.5). "
        "The `no_prediction` row is the zero-return forecast; its trading column is "
        "flat (cash), not buy-and-hold. Use the buy-and-hold figures in the results "
        "table for the trading comparison.",
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

    if cw_rows:
        lines += [
            "",
            "## Clark–West test against the zero forecast (one-sided)",
            "",
            "The zero forecast is the restricted model nested in any linear forecast; "
            "for the tree and LSTM baselines the nesting is approximate, so read their "
            "rows as a diagnostic rather than a formal nested test. The null is that "
            "every coefficient of the model is zero, and `p_value` is one-sided. The "
            "adjusted loss differential adds `(pred_zero − pred_model)²`, which does "
            "**not** make this an MSE-improvement test: the adjustment credits the "
            "forecast's own variance, so a positive statistic alongside a worse raw "
            "RMSE is the signature of an over-dispersed forecast (right sign, too "
            "large). Use the DM rows and the confidence set for accuracy ordering; "
            "use this table only for whether the forecast carries information.",
            "",
        ]
        lines.append(
            markdown_table(
                [
                    {**row,
                     "mean_loss_diff": f"{row['mean_loss_diff']:+.2e}",
                     "p_value": f"{row['p_value']:.4g}"}
                    for row in cw_rows
                ],
                ["model", "mean_loss_diff", "cw_stat", "p_value", "n", "outcome"],
            )
        )

    lines += [
        "",
        "## Model confidence set (range-based, α = 0.10)",
        "",
        "Hansen, Lunde and Nason (2011): models are eliminated one at a time, "
        "largest standardised excess loss first, until the range statistic falls "
        "below its stationary-bootstrap critical value. The surviving set contains "
        "every model that cannot be separated from the best at this level. "
        "Elimination order is data-driven; a wide set is the honest outcome for "
        "daily returns.",
        "",
        f"Included: {', '.join(f'`{name}`' for name in mcs['included'])}",
        "",
    ]
    if mcs["eliminated"]:
        lines.append(
            markdown_table(
                [
                    {**row, "tr": f"{row['tr']:.3f}", "critical": f"{row['critical']:.3f}"}
                    for row in mcs["eliminated"]
                ],
                ["step", "model", "tr", "critical"],
            )
        )
    else:
        lines.append("_No model was eliminated._")

    if mde_rows:
        lines += [
            "",
            "## Minimum detectable effect (power check)",
            "",
            "80%-power minimum detectable RMSE skill at the pooled sample size, "
            "from `scripts/detectable_skill.py`. An observed skill below the MDE "
            "cannot be reliably distinguished from zero in this sample.",
            "",
        ]
        lines.append(
            markdown_table(
                [
                    {**row,
                     "observed_rmse_skill": f"{row['observed_rmse_skill']:+.2%}",
                     "mde_rmse_skill": f"{row['mde_rmse_skill']:.2%}",
                     "power_at_observed": f"{row['power_at_observed']:.2f}"}
                    for row in mde_rows
                ],
                ["comparison", "observed_rmse_skill", "mde_rmse_skill",
                 "power_at_observed", "n"],
            )
        )

    lines += [
        "",
        "## Reading the table",
        "",
        "- `mean_loss_diff` is mean squared error of A minus that of B; a positive "
        "value means A is worse.",
        "- A significant DM result is evidence of a mean loss difference under "
        "the test assumptions. Comparisons are exploratory and not multiplicity-adjusted; "
        "the reference baseline is selected by pooled RMSE.",
        "- The Clark–West statistic is one-sided: positive rejects the null that all "
        "model coefficients are zero. It is not an MSE-improvement test — see the "
        "note above the table.",
        "- The model confidence set is the multiplicity-controlled summary: a model "
        "outside it is significantly worse than the best one, a model inside it is not "
        "distinguishable from the best at α = 0.10.",
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
    print(
        markdown_table(
            pooled_rows,
            ["model", "rmse", "rmse_zero", "rmse_skill", "da", "sharpe", "sharpe_95ci"],
        )
    )
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
