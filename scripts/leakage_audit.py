"""Leakage audit for the data, folds, scaler, and persisted predictions.

Run before the 11/13 hard gate:
    python scripts/leakage_audit.py

Writes docs/leakage_audit.md.  Any FAIL is a blocker for the main results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATE_COL, FEATURES, PREDS_DIR, TARGET_COL  # noqa: E402
from src.data import load_prices  # noqa: E402
from src.features import compute_features, fit_scaler  # noqa: E402
from src.splits import make_folds_from_frame  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(condition), detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    prices = load_prices("D05_SI")
    frame = compute_features(prices)
    folds = make_folds_from_frame(frame)
    label_dates = pd.DatetimeIndex(pd.to_datetime(frame["label_date"]))

    # ------------------------------------------------------------------
    # 1. Data integrity
    # ------------------------------------------------------------------
    check("dates are sorted and unique", frame[DATE_COL].is_monotonic_increasing and frame[DATE_COL].is_unique)
    check("every label date follows its feature date", (frame["label_date"] > frame[DATE_COL]).all())
    next_matches = (
        frame["label_date"].iloc[:-1].reset_index(drop=True)
        == frame[DATE_COL].iloc[1:].reset_index(drop=True)
    )
    check("label_date is the next available trading row", bool(next_matches.all()))
    check("no NaN in features or target", not frame[FEATURES + [TARGET_COL]].isna().any().any())

    # ------------------------------------------------------------------
    # 2. Fold boundaries
    # ------------------------------------------------------------------
    for fold in folds:
        train_max = label_dates[fold.train_idx].max()
        val_max = label_dates[fold.val_idx].max()
        test_min = label_dates[fold.test_idx].min()
        test_max = label_dates[fold.test_idx].max()
        select_max = label_dates[fold.select_train_idx].max()
        overlap = len(np.intersect1d(fold.train_idx, fold.test_idx))
        check(
            f"{fold.fold_id}: train ends before test",
            train_max < test_min,
            f"{train_max.date()} < {test_min.date()}",
        )
        check(
            f"{fold.fold_id}: validation ends before test",
            val_max < test_min,
            f"{val_max.date()} < {test_min.date()}",
        )
        check(
            f"{fold.fold_id}: selection train ends before validation",
            select_max < fold.val_start,
            f"{select_max.date()} < {fold.val_start.date()}",
        )
        check(f"{fold.fold_id}: test inside its window", test_min >= fold.test_start and test_max <= fold.test_end)
        check(f"{fold.fold_id}: no train/test index overlap", overlap == 0)

    # ------------------------------------------------------------------
    # 3. Feature causality on real prices
    # ------------------------------------------------------------------
    cut_date = pd.Timestamp("2020-01-02")
    perturbed = prices.copy()
    future = pd.to_datetime(perturbed[DATE_COL]) > cut_date
    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        perturbed.loc[future, col] = perturbed.loc[future, col] * 2.0
    changed = compute_features(perturbed)

    base = frame.set_index(DATE_COL)[FEATURES]
    moved = changed.set_index(DATE_COL)[FEATURES]
    common = base.index[base.index <= cut_date].intersection(moved.index)
    causal = np.allclose(
        base.loc[common].to_numpy(),
        moved.loc[common].to_numpy(),
        equal_nan=False,
    )
    check(
        "features before 2020-01-02 unchanged when future prices double",
        causal,
        f"{len(common)} rows compared",
    )

    # ------------------------------------------------------------------
    # 4. Scaler depends only on selection-train rows
    # ------------------------------------------------------------------
    fold = folds[0]
    select = frame.iloc[fold.select_train_idx]
    scaler_a = fit_scaler(select[FEATURES])

    perturbed_frame = frame.copy()
    perturbed_frame.loc[frame.index[fold.val_idx], FEATURES] += 100.0
    perturbed_frame.loc[frame.index[fold.test_idx], FEATURES] -= 100.0
    select_b = perturbed_frame.iloc[fold.select_train_idx]
    scaler_b = fit_scaler(select_b[FEATURES])

    scaler_equal = all(
        getattr(scaler_a, attr).equals(getattr(scaler_b, attr))
        for attr in ("lower", "upper", "mean", "std")
    )
    check(
        "scaler parameters unchanged when validation/test features are perturbed",
        scaler_equal,
        f"fold {fold.fold_id}",
    )

    # ------------------------------------------------------------------
    # 5. Persisted predictions align with fold test rows
    # ------------------------------------------------------------------
    checked = 0
    for path in sorted(PREDS_DIR.glob("*/*.csv")):
        fold_id = path.stem
        try:
            saved = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            check(f"{path.parent.name}/{fold_id}: readable", False, str(exc))
            continue
        fold = next((f for f in folds if f.fold_id == fold_id), None)
        if fold is None:
            check(f"{path.parent.name}/{fold_id}: known fold", False)
            continue

        expected_dates = (
            pd.to_datetime(frame.iloc[fold.test_idx]["label_date"])
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )
        dates_ok = saved["date"].tolist() == expected_dates
        y_ok = np.allclose(
            saved["y_true"].to_numpy(),
            frame.iloc[fold.test_idx][TARGET_COL].to_numpy(),
        )
        finite_ok = bool(np.isfinite(saved[["y_true", "y_pred"]].to_numpy()).all())
        check(
            f"{path.parent.name}/{fold_id}: dates and y align",
            dates_ok and y_ok,
            f"{len(saved)} rows",
        )
        check(f"{path.parent.name}/{fold_id}: finite predictions", finite_ok)
        checked += 1
    check("at least one prediction file audited", checked > 0, f"{checked} files")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    failures = [name for name, ok, _ in RESULTS if not ok]
    lines = ["# Leakage audit", "", f"Checks: {len(RESULTS)}; failures: {len(failures)}", ""]
    lines.append("| Check | Result | Detail |")
    lines.append("|---|---|---|")
    for name, ok, detail in RESULTS:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    if failures:
        lines.extend(["", "## Blockers", ""] + [f"- {name}" for name in failures])

    report = ROOT / "docs" / "leakage_audit.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{len(RESULTS)} checks, {len(failures)} failures")
    print(f"Wrote {report}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
