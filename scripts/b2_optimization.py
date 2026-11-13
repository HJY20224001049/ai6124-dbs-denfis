"""Validation-only incremental optimization around the B2 candidate.

B2 is the equal-weight average of three one-rule DENFIS models, each fitted on
one feature (``ma5``, ``boll_b``, ``ret_0``) plus the two ex-dividend dummies.
This script tests one change at a time around that base:

* component set: 2, 5 or 7 univariate rules;
* ex-dividend encoding: both dummies, post-ex-day only, or none;
* combination: mean, median, inverse-variance weights;
* training window: expanding, 3-year or 5-year rolling;
* output shrinkage: 0.75, 1.0 or 1.25;
* own-price decomposition: overnight gap and intraday components.

The target is unchanged: ``y_t = ln(AdjClose_{t+1} / AdjClose_t)``. Only the
selection-training window (labels through Y-2) and the validation year (Y-1)
are indexed. Test years and the 2026 holdout are never used.

Adoption rule, fixed before looking at the results: a variant replaces B2 only
if its mean validation skill improves by at least 0.05 percentage points and
its positive-fold count does not fall. Otherwise B2 stays.

Run:
    python scripts/b2_optimization.py
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
from validation_candidates import (  # noqa: E402
    _bootstrap_ci,
    _fit_predict,
    _skill,
)
from src.data import load_prices  # noqa: E402
from src.features import fit_scaler, transform_scaler  # noqa: E402
from src.models.denfis import DENFIS  # noqa: E402
from src.splits import make_folds_from_frame  # noqa: E402

OUT_DIR = ROOT / "results" / "analysis"
NOTE_PATH = ROOT / "docs" / "b2_optimization.md"
REFIT = 3.0
LAM = 1.0
MIN_GAIN = 0.0005  # 0.05 percentage points


def _fit_predict_var(
    frame: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, eval_idx: np.ndarray,
    columns: list[str],
) -> tuple[np.ndarray, float]:
    """Fit one rule and return the evaluation prediction plus in-sample variance."""
    scaler = fit_scaler(frame.iloc[train_idx][columns])
    X_train = transform_scaler(frame.iloc[train_idx][columns], scaler)
    X_eval = transform_scaler(frame.iloc[eval_idx][columns], scaler)
    model = DENFIS(dthr=REFIT, lam=LAM, max_rules=1)
    model.fit(X_train, y[train_idx])
    in_sample = model.predict(X_train)
    variance = float(np.var(y[train_idx] - in_sample, ddof=1))
    return model.predict(X_eval), max(variance, 1e-12)


def _add_decomposition(frame: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    adj_open = prices["Open"].astype(float) * prices["Adj Close"].astype(float) / prices["Close"].astype(float)
    extra = pd.DataFrame(
        {
            "Date": prices["Date"],
            "gap": np.log(adj_open / prices["Adj Close"].astype(float).shift(1)),
            "intraday": np.log(prices["Adj Close"].astype(float) / adj_open),
        }
    )
    return frame.merge(extra, on="Date", how="left")


def _rolling_idx(frame: pd.DataFrame, train_idx: np.ndarray, end: pd.Timestamp, years: int) -> np.ndarray:
    label_dates = pd.to_datetime(frame["label_date"])
    start = end - pd.DateOffset(years=years)
    mask = (label_dates.iloc[train_idx] >= start) & (label_dates.iloc[train_idx] <= end)
    return train_idx[mask]


def _predict(
    frame: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    components: list[list[str]],
    combine: str = "mean",
    shrink: float = 1.0,
) -> np.ndarray:
    if combine == "invvar":
        parts, variances = [], []
        for columns in components:
            pred, variance = _fit_predict_var(frame, y, train_idx, eval_idx, columns)
            parts.append(pred)
            variances.append(variance)
        weights = np.array([1.0 / value for value in variances])
        weights = weights / weights.sum()
        return shrink * np.average(np.vstack(parts), axis=0, weights=weights)
    parts = [_fit_predict(frame, y, train_idx, eval_idx, columns) for columns in components]
    stacked = np.vstack(parts)
    if combine == "median":
        return shrink * np.median(stacked, axis=0)
    return shrink * stacked.mean(axis=0)


def _predict_calibrated(
    frame: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    components: list[list[str]],
) -> np.ndarray:
    """Fit a scalar calibration on the last 20% of the selection window."""
    dates = pd.to_datetime(frame["label_date"]).iloc[train_idx]
    order = np.argsort(dates.to_numpy())
    ordered = train_idx[order]
    cut = int(len(ordered) * 0.8)
    inner_train, inner_cal = ordered[:cut], ordered[cut:]
    pred_cal = _predict(frame, y, inner_train, inner_cal, components)
    design = np.c_[np.ones(len(pred_cal)), pred_cal]
    coef, *_ = np.linalg.lstsq(design, y[inner_cal], rcond=None)
    pred_eval = _predict(frame, y, train_idx, eval_idx, components)
    return coef[0] + coef[1] * pred_eval


def _candidates() -> dict[str, dict]:
    base_components = [["ma5"] + EVENT_A, ["boll_b"] + EVENT_A, ["ret_0"] + EVENT_A]
    no_exdiv = [["ma5"], ["boll_b"], ["ret_0"]]
    post_only = [["ma5", "is_post_exdiv_1"], ["boll_b", "is_post_exdiv_1"], ["ret_0", "is_post_exdiv_1"]]
    candidates = {
        "B2_base": {"components": base_components, "combine": "mean", "shrink": 1.0},
        "B2_k2": {"components": [["ma5"] + EVENT_A, ["boll_b"] + EVENT_A], "combine": "mean", "shrink": 1.0},
        "B2_k5": {
            "components": [
                ["ma5"] + EVENT_A,
                ["boll_b"] + EVENT_A,
                ["ret_0"] + EVENT_A,
                ["ret_4"] + EVENT_A,
                ["ma20"] + EVENT_A,
            ],
            "combine": "mean",
            "shrink": 1.0,
        },
        "B2_k7": {
            "components": [
                ["ma5"] + EVENT_A,
                ["boll_b"] + EVENT_A,
                ["ret_0"] + EVENT_A,
                ["ret_4"] + EVENT_A,
                ["ma20"] + EVENT_A,
                ["macd"] + EVENT_A,
                ["rsi14"] + EVENT_A,
            ],
            "combine": "mean",
            "shrink": 1.0,
        },
        "B2_post_only": {"components": post_only, "combine": "mean", "shrink": 1.0},
        "B2_no_exdiv": {"components": no_exdiv, "combine": "mean", "shrink": 1.0},
        "B2_median": {"components": base_components, "combine": "median", "shrink": 1.0},
        "B2_invvar": {"components": base_components, "combine": "invvar", "shrink": 1.0},
        "B2_shrink075": {"components": base_components, "combine": "mean", "shrink": 0.75},
        "B2_shrink125": {"components": base_components, "combine": "mean", "shrink": 1.25},
        "B2_gap_intraday": {
            "components": [["ma5"] + EVENT_A, ["boll_b"] + EVENT_A, ["gap"] + EVENT_A, ["intraday"] + EVENT_A],
            "combine": "mean",
            "shrink": 1.0,
        },
        "B2_roll3": {"components": base_components, "combine": "mean", "shrink": 1.0, "window": 3},
        "B2_roll5": {"components": base_components, "combine": "mean", "shrink": 1.0, "window": 5},
        "B2_winsor": {"components": base_components, "combine": "mean", "shrink": 1.0, "winsor": True},
        "B2_calib": {"components": base_components, "combine": "mean", "shrink": 1.0, "calib": True},
        "B2_plus_ref": {"components": base_components, "combine": "mean", "shrink": 1.0, "plus_ref": True},
    }
    return candidates


def main() -> None:
    prices = load_prices("D05_SI")
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices.sort_values("Date").reset_index(drop=True)
    frame = _add_decomposition(_build_frame(prices), prices)
    frame = frame.dropna(subset=["gap", "intraday"]).reset_index(drop=True)
    folds = [
        fold for fold in make_folds_from_frame(frame) if fold.fold_id != "holdout"
    ]
    y = frame["y"].to_numpy(float)
    candidates = _candidates()

    rows, fold_rows = [], []
    for name, spec in candidates.items():
        y_parts, pred_parts, skills = [], [], []
        for fold in folds:
            train_idx = fold.select_train_idx
            if "window" in spec:
                train_idx = _rolling_idx(frame, train_idx, fold.select_train_end, spec["window"])
            if spec.get("calib"):
                pred = _predict_calibrated(
                    frame, y, train_idx, fold.val_idx, spec["components"]
                )
            elif spec.get("plus_ref"):
                base_pred = _predict(
                    frame,
                    y,
                    train_idx,
                    fold.val_idx,
                    spec["components"],
                    combine=spec["combine"],
                    shrink=spec["shrink"],
                )
                ref_pred = _fit_predict(
                    frame, y, train_idx, fold.val_idx, SUBSET + EVENT_A
                )
                pred = 0.5 * (base_pred + ref_pred)
            else:
                fit_y = y
                if spec.get("winsor"):
                    low, high = np.quantile(y[train_idx], [0.01, 0.99])
                    fit_y = y.copy()
                    fit_y[train_idx] = np.clip(y[train_idx], low, high)
                pred = _predict(
                    frame,
                    fit_y,
                    train_idx,
                    fold.val_idx,
                    spec["components"],
                    combine=spec["combine"],
                    shrink=spec["shrink"],
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

    summary = pd.DataFrame(rows).sort_values("mean_fold_skill", ascending=False)
    folds_df = pd.DataFrame(fold_rows)
    base = summary[summary["candidate"] == "B2_base"].iloc[0]
    summary["delta_vs_base"] = summary["mean_fold_skill"] - base["mean_fold_skill"]
    summary["adoptable"] = (
        (summary["delta_vs_base"] >= MIN_GAIN)
        & (summary["folds_positive"] >= base["folds_positive"])
        & (summary["candidate"] != "B2_base")
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "b2_optimization.csv", index=False)
    folds_df.to_csv(OUT_DIR / "b2_optimization_folds.csv", index=False)

    lines = [
        "| candidate | mean validation skill | delta vs B2 | median | folds > 0 | pooled skill | pooled 95% CI |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.mean_fold_skill:+.3%} | {row.delta_vs_base:+.3%} | "
            f"{row.median_fold_skill:+.3%} | {row.folds_positive}/9 | {row.pooled_skill:+.3%} | "
            f"[{row.ci_low:+.3%}, {row.ci_high:+.3%}] |"
        )
    table = "\n".join(lines)
    print(table)

    adoptable = summary[summary["adoptable"]]["candidate"].tolist()
    note = f"""# B2 incremental optimization (validation only)

Run by `scripts/b2_optimization.py`. B2 is the equal-weight average of three
one-rule DENFIS models on `ma5`, `boll_b`, `ret_0`, each with the two
ex-dividend dummies. The target is unchanged:
`y_t = ln(AdjClose_{{t+1}} / AdjClose_{{t}})`. Each candidate is fitted on the
selection-training window (labels through Y-2) and scored on the validation
year (Y-1); the test years and the 2026 holdout are never indexed.

Adoption rule, fixed before the run: a variant replaces B2 only if mean
validation skill improves by at least 0.05 pp and the positive-fold count does
not fall. With nine validation years and this many candidates, any gain is
still optimistic; the 2026 holdout remains the clean test.

{table}

## Adoptable under the rule

{', '.join(adoptable) if adoptable else 'None. B2 stays the candidate.'}

## Caveat

These are validation-year results, and the validation years have already been
used to choose the features and the ex-dividend encoding. The table is an
optimization guide, not a confirmation. Do not switch the frozen model on the
basis of any 2017-2025 test result.

## Reproduction

```bash
python scripts/b2_optimization.py
```
"""
    NOTE_PATH.write_text(note, encoding="utf-8")
    print(f"Saved {OUT_DIR / 'b2_optimization.csv'}")
    print(f"Saved {OUT_DIR / 'b2_optimization_folds.csv'}")
    print(f"Saved {NOTE_PATH}")


if __name__ == "__main__":
    main()
