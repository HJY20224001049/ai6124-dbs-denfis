"""B2 — the pre-registered final configuration, run through the shared CV runner.

B2 is the equal-weight average of three one-rule DENFIS models, each fitted on
one feature (``ma5``, ``boll_b``, ``ret_0``) plus the two ex-dividend dummies
``is_exdiv`` and ``is_post_exdiv_1``, with ``dthr=3.0``, ``lam=1.0`` and
``max_rules=1``.

Two scopes:

``--scope validation`` (default, allowed at any time)
    Fit on the selection-training window (labels through Y-2) and score the
    validation year Y-1 for all nine folds.  The test years and the 2026 holdout
    are never indexed.  Reproduces the B2 row of ``docs/validation_candidates.md``.

``--scope test`` / ``--scope holdout`` (only after the recorded freeze)
    Run the shared CV runner, which refits on train + validation, selects the
    signal threshold on validation Sharpe, and scores the test year or the 2026
    holdout.  Writes ``results/preds/denfis_b2/`` and ``results/metrics/`` and
    refreshes ``results/summary.csv``.

Examples
--------
python scripts/run_b2.py
python scripts/run_b2.py --scope test --after-freeze --folds all
python scripts/run_b2.py --scope holdout --after-freeze
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from config import fold_seed  # noqa: E402
from event_feature_pilot import EVENT_A, SUBSET, _build_frame  # noqa: E402
from src.data import load_prices  # noqa: E402
from src.evaluation.cv import run_model_on_folds  # noqa: E402
from src.features import fit_scaler, transform_scaler  # noqa: E402
from src.models.ensemble import DENFISEnsemble  # noqa: E402
from src.splits import make_folds_from_frame  # noqa: E402

COLUMNS = SUBSET + EVENT_A
COMPONENTS = [[index, len(SUBSET), len(SUBSET) + 1] for index in range(len(SUBSET))]
GRID = [{"dthr": 3.0, "lam": 1.0}]
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 123


def _factory(params, seed):
    return DENFISEnsemble(COMPONENTS, seed=seed, **params)


def _skill(y: np.ndarray, pred: np.ndarray) -> float:
    return 1.0 - np.sqrt(np.mean((y - pred) ** 2)) / np.sqrt(np.mean(y**2))


def _bootstrap_ci(y_parts, pred_parts):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    n = len(y_parts)
    for draw in range(BOOTSTRAP_DRAWS):
        picks = rng.integers(0, n, n)
        y = np.concatenate([y_parts[j] for j in picks])
        pred = np.concatenate([pred_parts[j] for j in picks])
        draws[draw] = _skill(y, pred)
    return tuple(float(v) for v in np.percentile(draws, [2.5, 97.5]))


def run_validation(frame: pd.DataFrame, folds) -> None:
    y = frame["y"].to_numpy(float)
    skills, y_parts, pred_parts = [], [], []
    for fold in folds:
        select = frame.iloc[fold.select_train_idx]
        val = frame.iloc[fold.val_idx]
        scaler = fit_scaler(select[COLUMNS])
        X_select = transform_scaler(select[COLUMNS], scaler)
        X_val = transform_scaler(val[COLUMNS], scaler)
        model = DENFISEnsemble(COMPONENTS, seed=fold_seed(fold.rep, fold.fold),
                               dthr=GRID[0]["dthr"], lam=GRID[0]["lam"])
        model.fit(X_select, y[fold.select_train_idx])
        pred = model.predict(X_val)
        skill = _skill(y[fold.val_idx], pred)
        skills.append(skill)
        y_parts.append(y[fold.val_idx])
        pred_parts.append(pred)
        print(f"{fold.fold_id}  validation skill {skill:+.3%}")

    pooled = _skill(np.concatenate(y_parts), np.concatenate(pred_parts))
    low, high = _bootstrap_ci(y_parts, pred_parts)
    print()
    print(f"mean validation skill : {np.mean(skills):+.3%}")
    print(f"median                : {np.median(skills):+.3%}")
    print(f"folds > 0             : {int(np.sum(np.array(skills) > 0))}/{len(skills)}")
    print(f"pooled skill          : {pooled:+.3%}")
    print(f"pooled 95% CI         : [{low:+.3%}, {high:+.3%}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("validation", "test", "holdout"),
                        default="validation")
    parser.add_argument("--folds", default="all",
                        help="Comma-separated fold ids for --scope test.")
    parser.add_argument("--after-freeze", action="store_true",
                        help="Required for test or holdout scope. Only pass this "
                             "after documenting the freeze in docs/freeze.md.")
    args = parser.parse_args()

    if args.scope != "validation" and not args.after_freeze:
        raise SystemExit(
            "Test and holdout runs are one-shot and only allowed after the recorded "
            "freeze. Document the freeze in docs/freeze.md, then pass --after-freeze."
        )

    prices = load_prices("D05_SI")
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices.sort_values("Date").reset_index(drop=True)
    frame = _build_frame(prices)
    folds = make_folds_from_frame(frame)
    regular = [fold for fold in folds if fold.fold_id != "holdout"]

    if args.scope == "validation":
        run_validation(frame, regular)
        return

    fold_ids = ["holdout"] if args.scope == "holdout" else (
        None if args.folds == "all" else [x.strip() for x in args.folds.split(",")]
    )
    summary = run_model_on_folds(
        "denfis_b2", _factory, folds, frame, GRID,
        fold_ids=fold_ids, columns=COLUMNS,
    )
    print()
    print(summary[summary["model"] == "denfis_b2"].to_string(index=False))


if __name__ == "__main__":
    main()
