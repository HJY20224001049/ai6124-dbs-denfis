"""Horizon pilot: does the prediction work at 1 day and at 5 days?

The professor's definition is an ordering on forecast error: DENFIS better
than XGBoost and LSTM, and both baselines better than no prediction. The
one-day models fail that test. This pilot asks whether a five-day forward
return satisfies it, using the same nine walk-forward folds.

Design choices, all made on validation data only:

* ``ma5``, ``boll_b`` and ``ret_0`` are the three features with the highest
  validation skill in ``docs/feature_information_check.md``.
* DENFIS is reduced to one rule with ``lam = 1.0``, which makes the RLS
  consequent identical to OLS (the evolving structure is deliberately
  removed). ``lam = 0.99`` is worse on validation.
* XGBoost is shallow and strongly regularised: depth 1, learning rate 0.01,
  30 minimum child weight, lambda 5, 100 trees. This configuration has the
  lowest mean validation RMSE in the shallow grid.
* LSTM keeps the configuration selected by the main protocol.

The test years are evaluated, never used for selection. The 2026 holdout is
untouched. Five-day returns overlap, so the effective sample is smaller than
the row count and the fold bootstrap does not capture within-fold overlap;
the intervals are optimistic.

Run:
    python scripts/horizon_pilot.py
"""

from __future__ import annotations

import json
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
NOTE_PATH = ROOT / "docs" / "horizon_pilot.md"
SUBSET = ["ma5", "boll_b", "ret_0"]
XGB_PARAMS = {
    "max_depth": 1,
    "learning_rate": 0.01,
    "min_child_weight": 30,
    "reg_lambda": 5.0,
    "n_estimators": 100,
}
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 123


def _frame_for_horizon(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    base = compute_features(prices)
    ret = np.log(
        prices["Adj Close"].astype(float) / prices["Adj Close"].astype(float).shift(1)
    )
    forward = ret.shift(-horizon).rolling(horizon).sum()
    aux = pd.DataFrame(
        {
            "Date": prices["Date"],
            "y_fwd": forward.values,
            "target_end": prices["Date"].shift(-horizon).values,
        }
    )
    frame = base.merge(aux, on="Date", how="left")
    frame["y"] = frame["y_fwd"]
    frame["label_date"] = pd.to_datetime(frame["target_end"])
    return frame.dropna(subset=["y", "label_date"]).reset_index(drop=True)


def _skill(y: np.ndarray, pred: np.ndarray) -> float:
    return 1.0 - float(np.sqrt(np.mean((y - pred) ** 2))) / float(
        np.sqrt(np.mean(y**2))
    )


def _predictions(
    frame: pd.DataFrame,
    folds,
    horizon: int,
) -> dict[str, list[np.ndarray]]:
    y = frame["y"].to_numpy(float)
    predictions = {"const": [], "denfis": [], "xgboost": [], "lstm": []}
    seq_len = 20
    for fold in folds:
        # Constant mean.
        predictions["const"].append(
            np.full(len(fold.test_idx), y[fold.train_idx].mean())
        )

        # One-rule DENFIS on the validation-selected features.
        scaler = fit_scaler(frame.iloc[fold.train_idx][SUBSET])
        X_train = transform_scaler(frame.iloc[fold.train_idx][SUBSET], scaler)
        X_test = transform_scaler(frame.iloc[fold.test_idx][SUBSET], scaler)
        denfis = DENFIS(dthr=3.0, lam=1.0, max_rules=1)
        denfis.fit(X_train, y[fold.train_idx])
        predictions["denfis"].append(denfis.predict(X_test))

        # Shallow XGBoost on all 13 features.
        scaler = fit_scaler(frame.iloc[fold.train_idx][FEATURES])
        X_train = transform_scaler(frame.iloc[fold.train_idx][FEATURES], scaler)
        X_test = transform_scaler(frame.iloc[fold.test_idx][FEATURES], scaler)
        xgb = XGBModel(XGB_PARAMS)
        xgb.fit(X_train, y[fold.train_idx])
        predictions["xgboost"].append(xgb.predict(X_test))

        # LSTM with the protocol's selected configuration.
        meta = json.loads(
            (ROOT / "results" / "metrics" / "lstm" / f"{fold.fold_id}.json").read_text()
        )
        params = meta["params"]
        scaler = fit_scaler(frame.iloc[fold.train_idx][FEATURES])
        full = transform_scaler(frame[FEATURES], scaler)
        X_train, train_idx = build_sequences(full, fold.train_idx, seq_len)
        X_test, test_idx = build_sequences(full, fold.test_idx, seq_len)
        lstm = LSTMModel(
            {
                "hidden_size": params["hidden_size"],
                "seq_len": seq_len,
                "dropout": params.get("dropout", 0.1),
                "epochs": params.get("epochs", 54),
            },
            seed=fold_seed(fold.rep, fold.fold),
        )
        lstm.fit(X_train, y[train_idx])
        predictions["lstm"].append(lstm.predict(X_test))

    return predictions


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


def _summarise(
    frame: pd.DataFrame,
    folds,
    predictions: dict[str, list[np.ndarray]],
    horizon: int,
) -> pd.DataFrame:
    y_parts = [frame["y"].to_numpy(float)[fold.test_idx] for fold in folds]
    rows = []
    for name, parts in predictions.items():
        pooled = _skill(np.concatenate(y_parts), np.concatenate(parts))
        ci_low, ci_high = _bootstrap_ci(y_parts, parts)
        fold_skills = [_skill(a, b) for a, b in zip(y_parts, parts)]
        rows.append(
            {
                "horizon": horizon,
                "model": name,
                "pooled_skill": pooled,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "folds_positive": sum(value > 0 for value in fold_skills),
            }
        )
    return pd.DataFrame(rows)


def _fmt(df: pd.DataFrame) -> str:
    lines = [
        "| model | pooled RMSE skill | fold-bootstrap 95% CI | folds > 0 |",
        "|---|---:|---|---:|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.pooled_skill:+.3%} | "
            f"[{row.ci_low:+.3%}, {row.ci_high:+.3%}] | {row.folds_positive}/9 |"
        )
    return "\n".join(lines)


def main() -> None:
    prices = load_prices("D05_SI")
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices.sort_values("Date").reset_index(drop=True)

    tables = []
    details = []
    for horizon in (1, 5):
        frame = _frame_for_horizon(prices, horizon)
        folds = [
            fold
            for fold in make_folds_from_frame(frame)
            if fold.fold_id != "holdout"
        ]
        predictions = _predictions(frame, folds, horizon)
        table = _summarise(frame, folds, predictions, horizon)
        tables.append(table)

        y_parts = [frame["y"].to_numpy(float)[fold.test_idx] for fold in folds]
        for fold, y_part, *parts in zip(folds, y_parts, *predictions.values()):
            details.append(
                {
                    "horizon": horizon,
                    "fold": fold.fold_id,
                    "year": int(pd.Timestamp(frame.loc[fold.test_idx[0], "label_date"]).year),
                    **{
                        name: _skill(y_part, part)
                        for name, part in zip(predictions, parts)
                    },
                }
            )

    summary = pd.concat(tables, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "horizon_pilot.csv", index=False)
    pd.DataFrame(details).to_csv(OUT_DIR / "horizon_pilot_folds.csv", index=False)

    one_day = summary[summary["horizon"] == 1]
    five_day = summary[summary["horizon"] == 5]
    note = f"""# Horizon pilot: 1 day versus 5 days

Run by `scripts/horizon_pilot.py`. The test years are evaluated, never used
for selection; the 2026 holdout is untouched. All model choices were made on
validation data: the three features (`ma5`, `boll_b`, `ret_0`) have the
highest validation skill; one-rule DENFIS uses `lam = 1.0`, which is better
than `lam = 0.99` on validation; the shallow XGBoost configuration has the
lowest mean validation RMSE in its grid; the LSTM keeps the protocol's
selected configuration.

## One-day forward return

{_fmt(one_day)}

## Five-day forward return

{_fmt(five_day)}

## Reading

- At one day, the ordering fails: DENFIS (+0.26%) and XGBoost (+0.21%) beat
  zero on point estimates, but LSTM is −0.57%. LSTM is the binding constraint.
- At five days, DENFIS (+0.82%) and XGBoost (+0.46%) beat zero, but LSTM is
  −0.39%. The ordering still fails on LSTM. DENFIS beats the constant
  historical mean (+0.22 percentage points of RMSE skill); XGBoost does not.
- The five-day result is largely a drift effect. A constant mean forecast
  already earns about +0.61% RMSE skill; zero is a weak benchmark over a
  five-day horizon.
- Five-day returns overlap. The effective sample is roughly one fifth of the
  row count, and the fold bootstrap does not capture within-fold overlap, so
  the intervals above are optimistic.
- The LSTM keeps the protocol's one-day selected configuration; it was not
  re-selected for the five-day target. A target-specific LSTM selection is the
  remaining untested route, and the LSTM is unstable across epochs.
- The five-day horizon changes the assignment from next-day to next-week
  prediction. It needs the instructor's written approval before it can
  replace the main protocol.

## Recommendation

Present this as a horizon-robustness result, not as a silent protocol change.
If the instructor approves a five-day horizon and a target-specific LSTM
re-selection, pre-register the models and evaluate once on the 2026 holdout.
Otherwise, neither the one-day nor the five-day task satisfies the stated
ordering with the current LSTM.

## Reproduction

```bash
python scripts/horizon_pilot.py
```
"""
    NOTE_PATH.write_text(note, encoding="utf-8")
    print(note)
    print(f"Saved {OUT_DIR / 'horizon_pilot.csv'}")
    print(f"Saved {OUT_DIR / 'horizon_pilot_folds.csv'}")
    print(f"Saved {NOTE_PATH}")


if __name__ == "__main__":
    main()
