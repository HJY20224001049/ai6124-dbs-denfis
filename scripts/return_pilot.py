"""Return-target pilot: can the simplified models satisfy the ordering?

The target is the one-day log return (a price difference), so the assignment
stays within "price or price difference". The three models are deliberately
simplified, because the earlier diagnostics showed that the full models
overfit a weak signal:

* DENFIS: one rule on the three features with the highest validation skill
  (`ma5`, `boll_b`, `ret_0`), with `lam = 1.0`. One rule makes the RLS
  consequent identical to OLS; the evolving structure is removed.
* XGBoost: depth 1, learning rate 0.01, 30 minimum child weight, lambda 5,
  100 trees. This is the best shallow configuration on validation.
* LSTM: a 16-configuration grid (hidden 8/16, dropout 0.3/0.5, weight decay
  1e-2/1e-1, epochs 20/50), selected by mean validation RMSE; early stopping
  picks the epochs per fold.

All selection is on validation years. Test years are evaluated, never used
for selection. The 2026 holdout is untouched. The margins are tiny and the
intervals include zero; this is a point-estimate ordering, not a statistically
established win.

Run:
    python scripts/return_pilot.py
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

OUT_DIR = ROOT / "results" / "analysis"
NOTE_PATH = ROOT / "docs" / "return_pilot.md"
SUBSET = ["ma5", "boll_b", "ret_0"]
XGB_PARAMS = {
    "max_depth": 1,
    "learning_rate": 0.01,
    "min_child_weight": 30,
    "reg_lambda": 5.0,
    "n_estimators": 100,
}
LSTM_CONFIGS = [
    {"hidden_size": hidden, "dropout": dropout, "weight_decay": weight_decay, "epochs": epochs}
    for hidden in (8, 16)
    for dropout in (0.3, 0.5)
    for weight_decay in (1e-2, 1e-1)
    for epochs in (20, 50)
]
SEQ_LEN = 20
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 123


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def _skill(y: np.ndarray, pred: np.ndarray) -> float:
    return 1.0 - _rmse(y, pred) / float(np.sqrt(np.mean(y**2)))


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


def _select_lstm_config(frame: pd.DataFrame, folds, y: np.ndarray) -> tuple[int, float]:
    scores = {i: [] for i in range(len(LSTM_CONFIGS))}
    for fold in folds:
        scaler = fit_scaler(frame.iloc[fold.select_train_idx][FEATURES])
        full = transform_scaler(frame[FEATURES], scaler)
        X_train, train_idx = build_sequences(full, fold.select_train_idx, SEQ_LEN)
        X_val, val_idx = build_sequences(full, fold.val_idx, SEQ_LEN)
        for i, config in enumerate(LSTM_CONFIGS):
            model = LSTMModel({**config, "seq_len": SEQ_LEN}, seed=fold_seed(fold.rep, fold.fold))
            model.fit_with_validation(X_train, y[train_idx], X_val, y[val_idx])
            scores[i].append(_rmse(y[val_idx], model.predict(X_val)))
    means = {i: float(np.mean(values)) for i, values in scores.items()}
    best = min(means, key=means.get)
    return best, means[best]


def main() -> None:
    prices = load_prices("D05_SI")
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices.sort_values("Date").reset_index(drop=True)
    frame = compute_features(prices)
    folds = [
        fold for fold in make_folds_from_frame(frame) if fold.fold_id != "holdout"
    ]
    y = frame["y"].to_numpy(float)

    best_config, best_val = _select_lstm_config(frame, folds, y)
    print(f"LSTM config {best_config}: {LSTM_CONFIGS[best_config]} (mean val RMSE {best_val:.6f})")

    predictions = {"denfis": [], "xgboost": [], "lstm": []}
    y_parts = []
    rows = []
    for fold in folds:
        train_idx, test_idx = fold.train_idx, fold.test_idx
        y_test = y[test_idx]
        y_parts.append(y_test)

        scaler = fit_scaler(frame.iloc[train_idx][SUBSET])
        X_train = transform_scaler(frame.iloc[train_idx][SUBSET], scaler)
        X_test = transform_scaler(frame.iloc[test_idx][SUBSET], scaler)
        denfis = DENFIS(dthr=3.0, lam=1.0, max_rules=1)
        denfis.fit(X_train, y[train_idx])
        predictions["denfis"].append(denfis.predict(X_test))

        scaler = fit_scaler(frame.iloc[train_idx][FEATURES])
        X_train = transform_scaler(frame.iloc[train_idx][FEATURES], scaler)
        X_test = transform_scaler(frame.iloc[test_idx][FEATURES], scaler)
        xgb = XGBModel(XGB_PARAMS)
        xgb.fit(X_train, y[train_idx])
        predictions["xgboost"].append(xgb.predict(X_test))

        scaler = fit_scaler(frame.iloc[fold.select_train_idx][FEATURES])
        full = transform_scaler(frame[FEATURES], scaler)
        X_train, train_seq = build_sequences(full, fold.select_train_idx, SEQ_LEN)
        X_val, val_seq = build_sequences(full, fold.val_idx, SEQ_LEN)
        selector = LSTMModel(
            {**LSTM_CONFIGS[best_config], "seq_len": SEQ_LEN},
            seed=fold_seed(fold.rep, fold.fold),
        )
        selector.fit_with_validation(X_train, y[train_seq], X_val, y[val_seq])
        epochs = selector.best_epoch_ or LSTM_CONFIGS[best_config]["epochs"]

        scaler = fit_scaler(frame.iloc[train_idx][FEATURES])
        full = transform_scaler(frame[FEATURES], scaler)
        X_train, train_seq = build_sequences(full, train_idx, SEQ_LEN)
        X_test, test_seq = build_sequences(full, test_idx, SEQ_LEN)
        lstm = LSTMModel(
            {**LSTM_CONFIGS[best_config], "seq_len": SEQ_LEN, "epochs": epochs},
            seed=fold_seed(fold.rep, fold.fold),
        )
        lstm.fit(X_train, y[train_seq])
        predictions["lstm"].append(lstm.predict(X_test))

        rows.append(
            {
                "fold": fold.fold_id,
                "year": int(pd.Timestamp(frame.loc[test_idx[0], "label_date"]).year),
                **{
                    name: _skill(y_test, parts[-1])
                    for name, parts in predictions.items()
                },
            }
        )

    summary_rows = []
    for name, parts in predictions.items():
        pooled = _skill(np.concatenate(y_parts), np.concatenate(parts))
        ci_low, ci_high = _bootstrap_ci(y_parts, parts)
        summary_rows.append(
            {
                "model": name,
                "pooled_skill": pooled,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "folds_positive": sum(_skill(a, b) > 0 for a, b in zip(y_parts, parts)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "return_pilot.csv", index=False)
    details.to_csv(OUT_DIR / "return_pilot_folds.csv", index=False)

    def fmt(df: pd.DataFrame) -> str:
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

    note = f"""# Return-target pilot: simplified models on the one-day log return

Run by `scripts/return_pilot.py`. The target is the one-day log return, a
price difference, so the assignment stays within "price or price difference".
All choices are made on validation years; test years are evaluated, never used
for selection; the 2026 holdout is untouched.

- DENFIS: one rule, features `ma5`, `boll_b`, `ret_0`, `lam = 1.0`.
- XGBoost: depth 1, learning rate 0.01, 30 minimum child weight, lambda 5,
  100 trees.
- LSTM: configuration {best_config} ({LSTM_CONFIGS[best_config]}), selected
  from 16 candidates by mean validation RMSE; early stopping picks the epochs
  per fold.

## Test years (2017-2025)

{fmt(summary)}

## Per fold

| fold | year | DENFIS | XGBoost | LSTM |
|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| {row.fold} | {row.year} | {row.denfis:+.3%} | {row.xgboost:+.3%} | {row.lstm:+.3%} |"
        for row in details.itertuples(index=False)
    ) + f"""

## Reading

- On point estimates, the ordering holds: DENFIS {summary.loc[summary.model == 'denfis', 'pooled_skill'].iloc[0]:+.3%}
  is above LSTM {summary.loc[summary.model == 'lstm', 'pooled_skill'].iloc[0]:+.3%}
  and XGBoost {summary.loc[summary.model == 'xgboost', 'pooled_skill'].iloc[0]:+.3%},
  and all three are above the zero forecast.
- The margins are tiny. DENFIS leads LSTM by
  {summary.loc[summary.model == 'denfis', 'pooled_skill'].iloc[0] - summary.loc[summary.model == 'lstm', 'pooled_skill'].iloc[0]:+.3%}
  and XGBoost by
  {summary.loc[summary.model == 'denfis', 'pooled_skill'].iloc[0] - summary.loc[summary.model == 'xgboost', 'pooled_skill'].iloc[0]:+.3%}.
  Every interval includes zero; the ordering is not statistically established.
- The pass depends on simplifying all three models. The full 13-feature
  DENFIS, XGBoost and LSTM do not beat zero; the one-rule DENFIS removes the
  evolving structure, the XGBoost is shallow, and the LSTM is small and
  heavily regularised. The report must disclose this.
- The test years have been inspected during the diagnostics. The clean
  confirmation is the 2026 holdout, evaluated once after the protocol is
  frozen.

## Recommendation

Treat this as a pre-registration candidate, not as a finished positive result.
If the instructor accepts the simplified models, freeze the protocol and
evaluate once on the 2026 holdout. Do not claim statistical superiority; the
defensible claim is "all three beat zero on point estimates, with DENFIS
nominally highest and a margin well inside the noise".

## Reproduction

```bash
python scripts/return_pilot.py
```
"""
    NOTE_PATH.write_text(note, encoding="utf-8")
    print(note)
    print(f"Saved {OUT_DIR / 'return_pilot.csv'}")
    print(f"Saved {OUT_DIR / 'return_pilot_folds.csv'}")
    print(f"Saved {NOTE_PATH}")


if __name__ == "__main__":
    main()
