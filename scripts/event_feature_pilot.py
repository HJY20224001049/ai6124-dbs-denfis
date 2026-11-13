"""Validation-only pilot: do event/calendar features help the models?

The target is the one-day log return. Event features are built only from data
available at the feature date:

* ``is_exdiv``: the date is an ex-dividend date, inferred from a positive
  ``Adj Close / Close`` jump above 0.5%. The ex-date is announced in advance,
  so the dummy is causal.
* ``is_post_exdiv_1``: the date is the first trading day after an ex-date.
* Calendar features: earnings month (Feb/May/Aug/Nov), month-end, month-start,
  Monday. These are known in advance.

Two feature sets are compared with the base features:

* A: ``is_exdiv`` + ``is_post_exdiv_1``.
* B: A plus earnings month, month-end and Monday.

Everything is selected on validation years only: fit on labels through Y-2,
score the validation year Y-1. The test years and the 2026 holdout are not
touched. After the validation grid, a descriptive check fits the same one-rule
DENFIS on train + validation and scores the test years; those numbers were
already inspected during the earlier diagnostics and are not used for any
selection.

Run:
    python scripts/event_feature_pilot.py
    python scripts/event_feature_pilot.py --test-only  # rebuild the note from the saved validation table
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

from config import FEATURES, fold_seed  # noqa: E402
from src.data import load_prices  # noqa: E402
from src.evaluation.cv import build_sequences  # noqa: E402
from src.features import compute_features, fit_scaler, transform_scaler  # noqa: E402
from src.models.baselines import LSTMModel, XGBModel  # noqa: E402
from src.models.denfis import DENFIS  # noqa: E402
from src.splits import make_folds_from_frame  # noqa: E402

OUT_DIR = ROOT / "results" / "analysis"
NOTE_PATH = ROOT / "docs" / "event_feature_pilot.md"
SUBSET = ["ma5", "boll_b", "ret_0"]
EVENT_A = ["is_exdiv", "is_post_exdiv_1"]
EVENT_B = EVENT_A + ["is_earnings_month", "is_month_end", "is_monday"]
XGB_PARAMS = {
    "max_depth": 1,
    "learning_rate": 0.01,
    "min_child_weight": 30,
    "reg_lambda": 5.0,
    "n_estimators": 100,
}
LSTM_CFG = {
    "hidden_size": 8,
    "dropout": 0.3,
    "weight_decay": 0.01,
    "epochs": 50,
    "seq_len": 20,
}
SEQ_LEN = 20
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 123


def _build_frame(prices: pd.DataFrame) -> pd.DataFrame:
    ratio = prices["Adj Close"].astype(float) / prices["Close"].astype(float)
    exdiv = set(prices.loc[ratio.pct_change() > 0.005, "Date"])
    calendar = pd.DataFrame({"Date": prices["Date"]})
    calendar["is_exdiv"] = calendar["Date"].isin(exdiv).astype(int)
    days_since = []
    counter = 99
    for date in calendar["Date"]:
        if date in exdiv:
            counter = 0
        else:
            counter = min(counter + 1, 10)
        days_since.append(counter)
    calendar["days_since_exdiv"] = days_since
    calendar["is_post_exdiv_1"] = (calendar["days_since_exdiv"] == 1).astype(int)
    calendar["is_earnings_month"] = calendar["Date"].dt.month.isin([2, 5, 8, 11]).astype(int)
    rank = calendar.groupby([calendar["Date"].dt.year, calendar["Date"].dt.month]).cumcount()
    size = calendar.groupby([calendar["Date"].dt.year, calendar["Date"].dt.month])["Date"].transform("count")
    calendar["is_month_end"] = (rank >= size - 2).astype(int)
    calendar["is_monday"] = (calendar["Date"].dt.dayofweek == 0).astype(int)
    return compute_features(prices).merge(calendar, on="Date", how="left")


def _validation_skill(frame: pd.DataFrame, folds, columns: list[str], model: str):
    y = frame["y"].to_numpy(float)
    values = []
    for fold in folds:
        train_idx, val_idx = fold.select_train_idx, fold.val_idx
        scaler = fit_scaler(frame.iloc[train_idx][columns])
        X_train = transform_scaler(frame.iloc[train_idx][columns], scaler)
        X_val = transform_scaler(frame.iloc[val_idx][columns], scaler)
        y_train, y_val = y[train_idx], y[val_idx]
        if model == "denfis":
            estimator = DENFIS(dthr=3.0, lam=1.0, max_rules=1)
        elif model == "xgboost":
            estimator = XGBModel(XGB_PARAMS)
        else:
            full = transform_scaler(frame[columns], scaler)
            X_train, train_seq = build_sequences(full, train_idx, SEQ_LEN)
            X_val, val_seq = build_sequences(full, val_idx, SEQ_LEN)
            y_train, y_val = y[train_seq], y[val_seq]
            estimator = LSTMModel(LSTM_CFG, seed=fold_seed(fold.rep, fold.fold))
            estimator.fit_with_validation(X_train, y_train, X_val, y_val)
            pred = estimator.predict(X_val)
            values.append(1.0 - np.sqrt(np.mean((y_val - pred) ** 2)) / np.sqrt(np.mean(y_val**2)))
            continue
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_val)
        values.append(1.0 - np.sqrt(np.mean((y_val - pred) ** 2)) / np.sqrt(np.mean(y_val**2)))
    return values


def _skill(y: np.ndarray, pred: np.ndarray) -> float:
    return 1.0 - np.sqrt(np.mean((y - pred) ** 2)) / np.sqrt(np.mean(y**2))


def _bootstrap_ci(
    y_parts: list[np.ndarray], pred_parts: list[np.ndarray]
) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    n = len(y_parts)
    for i in range(BOOTSTRAP_DRAWS):
        picks = rng.integers(0, n, n)
        y = np.concatenate([y_parts[j] for j in picks])
        pred = np.concatenate([pred_parts[j] for j in picks])
        draws[i] = _skill(y, pred)
    return tuple(float(v) for v in np.percentile(draws, [2.5, 97.5]))


def _test_skill(frame: pd.DataFrame, folds, columns: list[str]) -> dict:
    """Descriptive check: one-rule DENFIS fitted on train + validation."""
    y = frame["y"].to_numpy(float)
    y_parts, pred_parts = [], []
    positive = 0
    for fold in folds:
        train_idx, test_idx = fold.train_idx, fold.test_idx
        scaler = fit_scaler(frame.iloc[train_idx][columns])
        X_train = transform_scaler(frame.iloc[train_idx][columns], scaler)
        X_test = transform_scaler(frame.iloc[test_idx][columns], scaler)
        estimator = DENFIS(dthr=3.0, lam=1.0, max_rules=1)
        estimator.fit(X_train, y[train_idx])
        pred = estimator.predict(X_test)
        y_test = y[test_idx]
        y_parts.append(y_test)
        pred_parts.append(pred)
        positive += bool(_skill(y_test, pred) > 0)
    pooled = _skill(np.concatenate(y_parts), np.concatenate(pred_parts))
    ci_low, ci_high = _bootstrap_ci(y_parts, pred_parts)
    return {
        "test_skill": float(pooled),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "folds_positive": positive,
    }


def main(test_only: bool = False) -> None:
    prices = load_prices("D05_SI")
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices.sort_values("Date").reset_index(drop=True)
    frame = _build_frame(prices)
    folds = [
        fold for fold in make_folds_from_frame(frame) if fold.fold_id != "holdout"
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if test_only:
        table = pd.read_csv(OUT_DIR / "event_feature_pilot.csv")
    else:
        rows = []
        for model, base in [
            ("denfis", SUBSET),
            ("xgboost", FEATURES),
            ("lstm", FEATURES),
        ]:
            base_values = _validation_skill(frame, folds, base, model)
            base_mean = float(np.mean(base_values))
            rows.append(
                {
                    "model": model,
                    "feature_set": "base",
                    "mean_validation_skill": base_mean,
                    "folds_positive": sum(value > 0 for value in base_values),
                    "delta_vs_base": 0.0,
                }
            )
            for name, event in [("A_exdiv", EVENT_A), ("B_exdiv_calendar", EVENT_B)]:
                values = _validation_skill(frame, folds, base + event, model)
                mean = float(np.mean(values))
                rows.append(
                    {
                        "model": model,
                        "feature_set": name,
                        "mean_validation_skill": mean,
                        "folds_positive": sum(value > 0 for value in values),
                        "delta_vs_base": mean - base_mean,
                    }
                )
        table = pd.DataFrame(rows)
        table.to_csv(OUT_DIR / "event_feature_pilot.csv", index=False)

    test_rows = []
    for name, event in [
        ("base", []),
        ("A_exdiv", EVENT_A),
        ("B_exdiv_calendar", EVENT_B),
    ]:
        result = _test_skill(frame, folds, SUBSET + event)
        test_rows.append({"feature_set": name, **result})
    test_table = pd.DataFrame(test_rows)
    test_table.to_csv(OUT_DIR / "event_feature_pilot_test.csv", index=False)

    def fmt(model: str) -> str:
        subset = table[table["model"] == model]
        lines = [
            "| feature set | mean validation skill | folds > 0 | delta vs base |",
            "|---|---:|---:|---:|",
        ]
        for row in subset.itertuples(index=False):
            lines.append(
                f"| {row.feature_set} | {row.mean_validation_skill:+.3%} | "
                f"{row.folds_positive}/9 | {row.delta_vs_base:+.3%} |"
            )
        return "\n".join(lines)

    def fmt_test() -> str:
        lines = [
            "| feature set | test RMSE skill | fold-bootstrap 95% CI | folds > 0 |",
            "|---|---:|---|---:|",
        ]
        for row in test_table.itertuples(index=False):
            lines.append(
                f"| {row.feature_set} | {row.test_skill:+.3%} | "
                f"[{row.ci_low:+.3%}, {row.ci_high:+.3%}] | {row.folds_positive}/9 |"
            )
        return "\n".join(lines)

    note = f"""# Event/calendar feature pilot (validation only, plus a descriptive test)

Run by `scripts/event_feature_pilot.py`. Fit on labels through Y-2, score the
validation year Y-1. The test years and the 2026 holdout are not used for any
selection.

Ex-dividend dates are inferred from positive `Adj Close / Close` jumps above
0.5% (48 dates). `is_post_exdiv_1` marks the first trading day after an
ex-date. Feature set A is the two ex-dividend dummies; set B adds earnings
month, month-end and Monday.

## DENFIS (one rule, `ma5`, `boll_b`, `ret_0`)

{fmt('denfis')}

## XGBoost (shallow, 13 features)

{fmt('xgboost')}

## LSTM (small network, 13 features)

{fmt('lstm')}

## Descriptive test of the DENFIS feature sets

The same one-rule configuration was fitted on train + validation and scored on
the test years (2017-2025). The test years were already inspected during the
earlier diagnostics, so this is a descriptive check, not a clean confirmation.

{fmt_test()}

## Reading

- The ex-dividend features help **DENFIS only**: +0.212% to +0.432% mean
  validation skill, 7/9 folds positive. XGBoost is unchanged-to-slightly worse
  and LSTM is slightly worse.
- The univariate `is_post_exdiv_1` rule was +0.324% with 9/9 validation folds,
  but the model gain is smaller; the event sample is only 48 dates, so the
  estimate is fragile.
- Adding calendar features (set B) dilutes DENFIS. The ex-dividend pair is the
  only promising set.
- The descriptive test gives +0.510% (7/9) for the ex-dividend pair versus
  +0.264% (6/9) for the base features. The fold-bootstrap interval for the
  ex-dividend pair is wide and close to zero, and the test years were already
  seen during the diagnostics, so this is not a clean confirmation.
- LSTM is still higher on validation (+0.520% base) than DENFIS with the
  ex-dividend pair (+0.432%), so the point-estimate test ordering is not
  guaranteed by the validation data.
- If adopted, the ex-dividend features must be pre-registered and evaluated
  once on the 2026 holdout. Do not select them on 2017-2025.

## Reproduction

```bash
python scripts/event_feature_pilot.py
python scripts/event_feature_pilot.py --test-only
```
"""
    NOTE_PATH.write_text(note, encoding="utf-8")
    print(note)
    print(f"Saved {OUT_DIR / 'event_feature_pilot.csv'}")
    print(f"Saved {OUT_DIR / 'event_feature_pilot_test.csv'}")
    print(f"Saved {NOTE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="skip the validation grid and rebuild the note from the saved validation table",
    )
    main(test_only=parser.parse_args().test_only)
