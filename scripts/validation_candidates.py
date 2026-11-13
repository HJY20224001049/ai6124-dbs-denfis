"""Validation-only candidates for the original next-day close-to-close target.

The target is unchanged: ``y_t = ln(AdjClose_{t+1} / AdjClose_t)``. Every
number here is computed on the selection-training window (labels through Y-2)
and the validation year (Y-1). The test years and the 2026 holdout are never
indexed.

Candidates
----------
B1  one-rule DENFIS on ``ma5`` plus the two ex-dividend dummies.
B2  equal-weight average of three one-rule DENFIS models, each fitted on one
    feature (``ma5``, ``boll_b``, ``ret_0``) plus the ex-dividend dummies.

Reference rows use the three-feature subset with and without the dummies, and
``ma5`` alone, so the comparison is on one fixed pipeline.

Run:
    python scripts/validation_candidates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from event_feature_pilot import EVENT_A, SUBSET, _build_frame  # noqa: E402
from src.data import load_prices  # noqa: E402
from src.features import fit_scaler, transform_scaler  # noqa: E402
from src.models.denfis import DENFIS  # noqa: E402
from src.splits import make_folds_from_frame  # noqa: E402

OUT_DIR = ROOT / "results" / "analysis"
NOTE_PATH = ROOT / "docs" / "validation_candidates.md"
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 123


def _skill(y: np.ndarray, pred: np.ndarray) -> float:
    return 1.0 - np.sqrt(np.mean((y - pred) ** 2)) / np.sqrt(np.mean(y**2))


def _fit_predict(
    frame: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, eval_idx: np.ndarray,
    columns: list[str],
) -> np.ndarray:
    scaler = fit_scaler(frame.iloc[train_idx][columns])
    X_train = transform_scaler(frame.iloc[train_idx][columns], scaler)
    X_eval = transform_scaler(frame.iloc[eval_idx][columns], scaler)
    model = DENFIS(dthr=3.0, lam=1.0, max_rules=1)
    model.fit(X_train, y[train_idx])
    return model.predict(X_eval)


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


def _candidate_predictions(
    frame: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    name: str,
) -> np.ndarray:
    if name == "B1_ma5_exdiv":
        return _fit_predict(frame, y, train_idx, eval_idx, ["ma5"] + EVENT_A)
    if name == "B1_ma5_only":
        return _fit_predict(frame, y, train_idx, eval_idx, ["ma5"])
    if name in ("B2_equal3_exdiv", "B2_equal3_base"):
        extra = EVENT_A if name.endswith("exdiv") else []
        parts = [
            _fit_predict(frame, y, train_idx, eval_idx, [feature] + extra)
            for feature in SUBSET
        ]
        return np.mean(parts, axis=0)
    if name == "ref_3feat_exdiv":
        return _fit_predict(frame, y, train_idx, eval_idx, SUBSET + EVENT_A)
    if name == "ref_3feat_base":
        return _fit_predict(frame, y, train_idx, eval_idx, SUBSET)
    raise ValueError(name)


def main() -> None:
    prices = load_prices("D05_SI")
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices.sort_values("Date").reset_index(drop=True)
    frame = _build_frame(prices)
    folds = [
        fold for fold in make_folds_from_frame(frame) if fold.fold_id != "holdout"
    ]
    y = frame["y"].to_numpy(float)

    names = [
        "B1_ma5_only",
        "B1_ma5_exdiv",
        "B2_equal3_base",
        "B2_equal3_exdiv",
        "ref_3feat_base",
        "ref_3feat_exdiv",
    ]
    rows = []
    fold_rows = []
    for name in names:
        y_parts, pred_parts, skills = [], [], []
        for fold in folds:
            # Only the selection-training and validation windows are indexed.
            pred = _candidate_predictions(
                frame, y, fold.select_train_idx, fold.val_idx, name
            )
            y_val = y[fold.val_idx]
            y_parts.append(y_val)
            pred_parts.append(pred)
            skill = _skill(y_val, pred)
            skills.append(skill)
            fold_rows.append({"candidate": name, "fold": fold.fold_id, "val_year": fold.test_year - 1, "skill": skill})
        pooled = _skill(np.concatenate(y_parts), np.concatenate(pred_parts))
        ci_low, ci_high = _bootstrap_ci(y_parts, pred_parts)
        rows.append(
            {
                "candidate": name,
                "mean_fold_skill": float(np.mean(skills)),
                "median_fold_skill": float(np.median(skills)),
                "folds_positive": sum(value > 0 for value in skills),
                "pooled_skill": pooled,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    summary = pd.DataFrame(rows)
    folds_df = pd.DataFrame(fold_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "validation_candidates.csv", index=False)
    folds_df.to_csv(OUT_DIR / "validation_candidates_folds.csv", index=False)

    lines = [
        "| candidate | mean validation skill | median | folds > 0 | pooled skill | pooled 95% CI |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.mean_fold_skill:+.3%} | {row.median_fold_skill:+.3%} | "
            f"{row.folds_positive}/9 | {row.pooled_skill:+.3%} | "
            f"[{row.ci_low:+.3%}, {row.ci_high:+.3%}] |"
        )
    table = "\n".join(lines)
    print(table)

    note = f"""# Validation-only candidates (original target)

Run by `scripts/validation_candidates.py`. The target is unchanged:
`y_t = ln(AdjClose_{{t+1}} / AdjClose_{{t}})`. Each model is fitted on the
selection-training window (labels through Y-2) and scored on the validation
year (Y-1). The test years and the 2026 holdout are never indexed.

- B1: one-rule DENFIS on `ma5` plus `is_exdiv`, `is_post_exdiv_1`.
- B2: equal-weight average of three one-rule DENFIS models, each on one of
  `ma5`, `boll_b`, `ret_0`, plus the ex-dividend dummies.
- Reference rows: `ma5` alone, the three-feature subset with/without dummies.

{table}

## Reading

The validation set is the only selection window here. The comparison with the
LSTM is against its validation-selected configuration (+0.520% mean validation
skill, 8/9 folds), which is itself a maximum over 16 candidates and therefore
optimistic. A candidate is only worth freezing if it improves on the current
three-feature + ex-dividend variant without relying on the test years.

## Reproduction

```bash
python scripts/validation_candidates.py
```
"""
    NOTE_PATH.write_text(note, encoding="utf-8")
    print(f"Saved {OUT_DIR / 'validation_candidates.csv'}")
    print(f"Saved {OUT_DIR / 'validation_candidates_folds.csv'}")
    print(f"Saved {NOTE_PATH}")


if __name__ == "__main__":
    main()
