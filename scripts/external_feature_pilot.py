"""Validation-only pilot: overnight US financials, USD/SGD and VIX.

The external series are cached under ``data/external/`` by
``scripts/download_external.py``. For each Singapore trading date ``t`` we use
the last US/FRED observation strictly before ``t`` (``merge_asof`` with
``allow_exact_matches=False``), so the feature is known before the Singapore
session opens.

Feature sets (each 1-3 features):

* S1: XLF overnight return.
* S2: BAC overnight return.
* S3: BAC return + USD/SGD log change.
* S4: BAC return + USD/SGD change + VIX log change.
* S5: VIX log change.
* S6: JPM overnight return.

The simplified models are the same as `docs/return_pilot.md`: one-rule DENFIS
on `ma5`/`boll_b`/`ret_0`, shallow XGBoost on 13 features, small LSTM on 13
features. Everything is scored on validation years only; the test years and
the 2026 holdout are untouched.

Run:
    python scripts/download_external.py
    python scripts/external_feature_pilot.py
"""

from __future__ import annotations

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

EXT_DIR = ROOT / "data" / "external"
OUT_DIR = ROOT / "results" / "analysis"
NOTE_PATH = ROOT / "docs" / "external_feature_pilot.md"
SUBSET = ["ma5", "boll_b", "ret_0"]
SETS = {
    "S1_xlf": ["xlf_ret"],
    "S2_bac": ["bac_ret"],
    "S3_bac_fx": ["bac_ret", "usdsgd_chg"],
    "S4_bac_fx_vix": ["bac_ret", "usdsgd_chg", "vix_chg"],
    "S5_vix": ["vix_chg"],
    "S6_jpm": ["jpm_ret"],
}
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


def _load_us(symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(EXT_DIR / f"{symbol}.csv", parse_dates=["Date"]).sort_values("Date")
    close = frame["Adj Close"].astype(float)
    frame["ret"] = np.log(close / close.shift(1))
    return frame[["Date", "ret"]].dropna().rename(
        columns={"Date": "ext_date", "ret": f"{symbol.lower()}_ret"}
    )


def _load_fred(name: str, column: str) -> pd.DataFrame:
    frame = pd.read_csv(EXT_DIR / f"{name}.csv", parse_dates=["observation_date"])
    frame = frame.rename(columns={"observation_date": "ext_date", name: column})
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna().sort_values("ext_date")


def _build_frame(prices: pd.DataFrame) -> pd.DataFrame:
    frame = compute_features(prices)
    external = (
        _load_us("JPM")
        .merge(_load_us("BAC"), on="ext_date")
        .merge(_load_us("XLF"), on="ext_date")
    )
    vix = _load_fred("VIXCLS", "vix")
    vix["vix_chg"] = np.log(vix["vix"] / vix["vix"].shift(1))
    fx = _load_fred("DEXSIUS", "usdsgd")
    fx["usdsgd_chg"] = np.log(fx["usdsgd"] / fx["usdsgd"].shift(1))
    external = (
        external.merge(vix[["ext_date", "vix_chg"]], on="ext_date")
        .merge(fx[["ext_date", "usdsgd_chg"]], on="ext_date")
        .dropna()
    )
    merged = pd.merge_asof(
        frame.sort_values("Date"),
        external.sort_values("ext_date"),
        left_on="Date",
        right_on="ext_date",
        direction="backward",
        allow_exact_matches=False,
    )
    return merged


def _validation_skill(frame, folds, columns, model):
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


def main() -> None:
    prices = load_prices("D05_SI")
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices.sort_values("Date").reset_index(drop=True)
    frame = _build_frame(prices)
    folds = [
        fold for fold in make_folds_from_frame(frame) if fold.fold_id != "holdout"
    ]

    rows = []
    for model, base in [("denfis", SUBSET), ("xgboost", FEATURES)]:
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
        for name, event in SETS.items():
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

    # LSTM only for S5, the only set that improved the other two models.
    base_values = _validation_skill(frame, folds, FEATURES, "lstm")
    base_mean = float(np.mean(base_values))
    rows.append(
        {
            "model": "lstm",
            "feature_set": "base",
            "mean_validation_skill": base_mean,
            "folds_positive": sum(value > 0 for value in base_values),
            "delta_vs_base": 0.0,
        }
    )
    values = _validation_skill(frame, folds, FEATURES + SETS["S5_vix"], "lstm")
    mean = float(np.mean(values))
    rows.append(
        {
            "model": "lstm",
            "feature_set": "S5_vix",
            "mean_validation_skill": mean,
            "folds_positive": sum(value > 0 for value in values),
            "delta_vs_base": mean - base_mean,
        }
    )

    table = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "external_feature_pilot.csv", index=False)

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

    note = f"""# Overnight external-feature pilot (validation only)

Run by `scripts/external_feature_pilot.py`; data cached by
`scripts/download_external.py`. For each Singapore date, the last US/FRED
observation strictly before it is used, so the features are known at the
Singapore open. Validation years only; the test years and 2026 holdout are
untouched.

## DENFIS (one rule, `ma5`, `boll_b`, `ret_0`)

{fmt('denfis')}

## XGBoost (shallow, 13 features)

{fmt('xgboost')}

## LSTM (small network, 13 features)

{fmt('lstm')}

## Reading

- No feature set improves all three models. `vix_chg` is the only set that
  improves DENFIS (+0.060 pp) and XGBoost (+0.001 pp), but it destroys LSTM
  (+0.520% to -0.001%).
- The univariate validation signals were positive (`bac_ret` +0.235%, `vix_chg`
  +0.297%), but adding them to the models does not translate into a joint
  improvement. This is the same overfitting pattern as the technical features.
- FOMC historical dates were not available in the accessible public CSV (only
  future 2026-2027 meetings); earnings dates were approximated by the earnings
  month in `docs/event_feature_pilot.md`, where they diluted the signal.
- Per the rule "if all three models do not improve on validation, stop", this
  direction is closed. Do not evaluate it on the test years or the 2026
  holdout.

## Reproduction

```bash
python scripts/download_external.py
python scripts/external_feature_pilot.py
```
"""
    NOTE_PATH.write_text(note, encoding="utf-8")
    print(note)
    print(f"Saved {OUT_DIR / 'external_feature_pilot.csv'}")
    print(f"Saved {NOTE_PATH}")


if __name__ == "__main__":
    main()
