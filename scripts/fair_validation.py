"""Fair validation scores: no model is selected on the year it is scored.

The protocol runner lets a model whose class defines ``fit_with_validation``
early-stop on the *outer* validation year and then scores that same year
(``src/evaluation/cv.py``).  Both XGBoost and the LSTM do this, so their
validation numbers are optimistic relative to B2, whose parameters are fixed
and never chosen on the scored year.

This script removes the advantage.  For each fold the selection-training window
is split again in time: its last year (Y-2) becomes an inner validation year,
everything before it (through Y-3) is inner training.  Hyperparameters and the
number of trees/epochs are chosen on that inner split only; the outer validation
year (Y-1) is predicted once and scored.

Validation years only; the test years and the 2026 holdout are never indexed.

Run:
    python scripts/fair_validation.py
    python scripts/fair_validation.py --models xgboost --folds r1f1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FEATURES, LSTM_CONFIGS, XGB_GRID, fold_seed  # noqa: E402
from src.data import load_prices  # noqa: E402
from src.evaluation.cv import build_sequences  # noqa: E402
from src.features import compute_features, fit_scaler, transform_scaler  # noqa: E402
from src.models.baselines import LSTMModel, XGBModel  # noqa: E402
from src.splits import make_folds_from_frame  # noqa: E402

SEQ_LEN = 20
OUT_DIR = ROOT / "results" / "analysis"
NOTE_PATH = ROOT / "docs" / "fair_validation.md"


def _skill(y: np.ndarray, pred: np.ndarray) -> float:
    return 1.0 - np.sqrt(np.mean((y - pred) ** 2)) / np.sqrt(np.mean(y**2))


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def _inner_split(label_dates: pd.Series, select_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Last calendar year of the selection window is the inner validation year."""
    years = label_dates.iloc[select_idx].dt.year.to_numpy()
    last = int(years.max())
    return select_idx[years < last], select_idx[years == last]


def _tabular_targets(frame, scaler, idx):
    X = transform_scaler(frame.iloc[idx][FEATURES], scaler)
    y = frame["y"].to_numpy(float)[idx]
    return X, y


def _run_xgboost(frame, fold, scaler, inner_train, inner_val):
    X_inner, y_inner = _tabular_targets(frame, scaler, inner_train)
    X_inner_val, y_inner_val = _tabular_targets(frame, scaler, inner_val)

    best = None
    for params in XGB_GRID:
        model = XGBModel(params, seed=fold_seed(fold.rep, fold.fold))
        model.fit_with_validation(X_inner, y_inner, X_inner_val, y_inner_val)
        score = _rmse(y_inner_val, model.predict(X_inner_val))
        if best is None or score < best["inner_rmse"]:
            best = {"inner_rmse": score, "params": dict(params),
                    "n_estimators": int(model.best_iteration_ or 0) + 1}

    final = XGBModel({**best["params"], "n_estimators": best["n_estimators"]},
                     seed=fold_seed(fold.rep, fold.fold))
    X_select, y_select = _tabular_targets(frame, scaler, fold.select_train_idx)
    final.fit(X_select, y_select)
    X_val, y_val = _tabular_targets(frame, scaler, fold.val_idx)
    pred = final.predict(X_val)
    return {
        "model": "xgboost",
        "fold_id": fold.fold_id,
        "inner_rmse": best["inner_rmse"],
        "epochs_or_trees": best["n_estimators"],
        "val_skill": _skill(y_val, pred),
        "y_val": y_val,
        "pred": pred,
    }


def _run_lstm(frame, fold, scaler, inner_train, inner_val):
    full = transform_scaler(frame[FEATURES], scaler)
    X_inner, inner_train_valid = build_sequences(full, inner_train, SEQ_LEN)
    y_inner = frame["y"].to_numpy(float)[inner_train_valid]
    X_inner_val, inner_val_valid = build_sequences(full, inner_val, SEQ_LEN)
    y_inner_val = frame["y"].to_numpy(float)[inner_val_valid]

    best = None
    for config in LSTM_CONFIGS:
        model = LSTMModel(config, seed=fold_seed(fold.rep, fold.fold))
        model.fit_with_validation(X_inner, y_inner, X_inner_val, y_inner_val)
        score = _rmse(y_inner_val, model.predict(X_inner_val))
        if best is None or score < best["inner_rmse"]:
            best = {"inner_rmse": score, "config": dict(config),
                    "epochs": int(model.best_epoch_ or model.epochs)}

    model = LSTMModel({**best["config"], "epochs": best["epochs"]},
                      seed=fold_seed(fold.rep, fold.fold))
    X_select, select_valid = build_sequences(full, fold.select_train_idx, SEQ_LEN)
    model.fit(X_select, frame["y"].to_numpy(float)[select_valid])
    X_val, val_valid = build_sequences(full, fold.val_idx, SEQ_LEN)
    y_val = frame["y"].to_numpy(float)[val_valid]
    pred = model.predict(X_val)
    return {
        "model": "lstm",
        "fold_id": fold.fold_id,
        "inner_rmse": best["inner_rmse"],
        "epochs_or_trees": best["epochs"],
        "val_skill": _skill(y_val, pred),
        "y_val": y_val,
        "pred": pred,
    }


RUNNERS = {"xgboost": _run_xgboost, "lstm": _run_lstm}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="xgboost,lstm")
    parser.add_argument("--folds", default="all")
    args = parser.parse_args()

    prices = load_prices("D05_SI")
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices.sort_values("Date").reset_index(drop=True)
    frame = compute_features(prices)
    label_dates = pd.to_datetime(frame["label_date"])
    folds = [fold for fold in make_folds_from_frame(frame) if fold.fold_id != "holdout"]
    if args.folds != "all":
        wanted = {x.strip() for x in args.folds.split(",")}
        folds = [fold for fold in folds if fold.fold_id in wanted]

    rows, pooled = [], {}
    for name in [x.strip() for x in args.models.split(",")]:
        runner = RUNNERS[name]
        y_parts, pred_parts = [], []
        for fold in folds:
            started = time.perf_counter()
            scaler = fit_scaler(frame.iloc[fold.select_train_idx][FEATURES])
            inner_train, inner_val = _inner_split(label_dates, fold.select_train_idx)
            result = runner(frame, fold, scaler, inner_train, inner_val)
            rows.append({key: result[key] for key in
                         ("model", "fold_id", "inner_rmse", "epochs_or_trees", "val_skill")})
            y_parts.append(result["y_val"])
            pred_parts.append(result["pred"])
            print(f"{name:8s} {fold.fold_id}  skill {result['val_skill']:+.3%}  "
                  f"inner {result['inner_rmse']:.6f}  "
                  f"({time.perf_counter() - started:.0f}s)")
        pooled[name] = _skill(np.concatenate(y_parts), np.concatenate(pred_parts))

    table = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "fair_validation.csv", index=False)

    print()
    summary = []
    for name, part in table.groupby("model"):
        summary.append({
            "model": name,
            "mean_skill": part["val_skill"].mean(),
            "median_skill": part["val_skill"].median(),
            "folds_positive": int((part["val_skill"] > 0).sum()),
            "n_folds": len(part),
            "pooled_skill": pooled[name],
        })
        print(f"{name:8s} mean {part['val_skill'].mean():+.3%}  "
              f"median {part['val_skill'].median():+.3%}  "
              f"folds>0 {int((part['val_skill'] > 0).sum())}/{len(part)}  "
              f"pooled {pooled[name]:+.3%}")

    NOTE_PATH.write_text(_note(pd.DataFrame(summary)), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'fair_validation.csv'}")
    print(f"Wrote {NOTE_PATH}")


def _note(summary: pd.DataFrame) -> str:
    lines = [
        "# Fair validation scores (no selection on the scored year)",
        "",
        "Both baselines early-stop on the outer validation year in the protocol "
        "runner, so their reported validation numbers are optimistic relative to "
        "B2 (fixed parameters). Here the selection-training window is split in "
        "time: its last year (Y-2) selects the configuration and the number of "
        "trees/epochs, and the outer validation year (Y-1) is scored once.",
        "",
        "| model | mean skill | median | folds > 0 | pooled skill |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['model']} | {row['mean_skill']:+.3%} | {row['median_skill']:+.3%} | "
            f"{int(row['folds_positive'])}/{int(row['n_folds'])} | {row['pooled_skill']:+.3%} |"
        )
    lines += [
        "",
        "## Reference point",
        "",
        "B2 (three one-rule DENFIS components, parameters fixed): mean +0.497%, "
        "8/9 folds, pooled +0.480% (`docs/validation_candidates.md`, "
        "`scripts/run_b2.py`).",
        "",
        "Ordering on mean validation skill: **B2 (+0.497%) > XGBoost (+0.021%) "
        "> LSTM (−1.237%) > zero**. The same ordering holds on the pooled "
        "validation years, although the XGBoost pooled value is slightly below "
        "zero (−0.029%).",
        "",
        "## Reading",
        "",
        "- The protocol runner early-stops both baselines on the year it then "
        "scores. That is why the validation numbers reported earlier (for "
        "example +0.520% for the LSTM) were higher: the epoch or tree count was "
        "chosen on the scored year.",
        "- This procedure is deliberately conservative for the baselines: the "
        "inner window is one year shorter, and two folds select a single tree, "
        "which is a symptom of a short inner window rather than of a useless "
        "model. Treat these numbers as lower bounds, not as the best the "
        "baselines can do.",
        "- B2 is not free of selection either: its features, ex-dividend dummies "
        "and hyperparameters were chosen on validation years, and +0.497% is the "
        "best of sixteen pre-registered variants. The comparison is "
        "validation-against-validation; the 2026 holdout is the only clean "
        "confirmation.",
        "- No test-year or holdout number appears here; none was computed.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/fair_validation.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
