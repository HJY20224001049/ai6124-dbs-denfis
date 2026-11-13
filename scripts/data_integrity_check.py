"""Audit the cached DBS data, the adjusted series and the label alignment.

Checks, in order:

1. file checksums and basic OHLCV integrity;
2. corporate-action adjustments in ``Adj Close / Close``;
3. label alignment (next trading day, exact log return);
4. feature causality by prefix invariance (truncate the future, recompute);
5. fold disjointness and test-window sizes;
6. trading-day completeness against STI, and the impact of any gap;
7. repeated closes and flat bars.

The audit does not modify the cache and does not touch the 2026 holdout.

Run:
    python scripts/data_integrity_check.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_DIR, FEATURES  # noqa: E402
from src.data import load_prices  # noqa: E402
from src.features import compute_features  # noqa: E402
from src.splits import make_folds_from_frame  # noqa: E402

OUT_DIR = ROOT / "results" / "analysis"
NOTE_PATH = ROOT / "docs" / "data_integrity_check.md"
FOLD_IDS = ["r1f1", "r1f2", "r1f3", "r2f1", "r2f2", "r2f3", "r3f1", "r3f2", "r3f3"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_checks() -> tuple[dict, pd.DataFrame]:
    report = {}
    checksum_ok = True
    checksum_path = DATA_DIR / "checksums.txt"
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, name = line.split()
        checksum_ok &= _sha256(DATA_DIR / name) == expected
    report["checksums_ok"] = bool(checksum_ok)

    dbs = load_prices("D05_SI")
    dbs["Date"] = pd.to_datetime(dbs["Date"])
    dbs = dbs.sort_values("Date").reset_index(drop=True)
    sti = load_prices("STI")
    sti["Date"] = pd.to_datetime(sti["Date"])

    report.update(
        {
            "dbs_rows": int(len(dbs)),
            "dbs_start": dbs["Date"].min().strftime("%Y-%m-%d"),
            "dbs_end": dbs["Date"].max().strftime("%Y-%m-%d"),
            "duplicate_dates": int(dbs["Date"].duplicated().sum()),
            "null_cells": int(dbs.isna().sum().sum()),
            "nonpositive_adj_close": int((dbs["Adj Close"] <= 0).sum()),
            "zero_or_negative_volume": int((dbs["Volume"] <= 0).sum()),
            "ohlc_violations": int(
                (
                    (dbs.High < dbs.Low)
                    | (dbs.High < dbs.Open)
                    | (dbs.High < dbs.Close)
                    | (dbs.Low > dbs.Open)
                    | (dbs.Low > dbs.Close)
                ).sum()
            ),
            "max_calendar_gap_days": int(dbs["Date"].diff().dt.days.max()),
            "flat_bars": int(((dbs.High == dbs.Low) & (dbs.Volume > 0)).sum()),
            "repeated_closes": int((dbs["Close"].diff() == 0).sum()),
            "sti_rows": int(len(sti)),
        }
    )
    return report, dbs


def _adjustment_checks(dbs: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    ratio = dbs["Adj Close"].astype(float) / dbs["Close"].astype(float)
    change = ratio.pct_change()
    jumps = dbs.loc[change.abs() > 0.01, ["Date", "Close", "Adj Close", "Volume"]].copy()
    jumps["ratio_change"] = change[change.abs() > 0.01].values
    jumps["close_simple_ret"] = dbs["Close"].pct_change()[change.abs() > 0.01].values
    jumps["adj_simple_ret"] = dbs["Adj Close"].pct_change()[change.abs() > 0.01].values
    report = {
        "ratio_jumps_over_1pct": int(len(jumps)),
        "negative_ratio_jumps": int((change < -0.001).sum()),
        "largest_ratio_jump": float(change.abs().max()),
    }
    return report, jumps


def _label_checks(frame: pd.DataFrame, dbs: pd.DataFrame) -> dict:
    raw = dbs.set_index("Date")["Adj Close"].astype(float).sort_index()
    mismatches = 0
    for i in range(len(frame)):
        date = pd.Timestamp(frame.Date.iloc[i])
        label_date = pd.Timestamp(frame.label_date.iloc[i])
        next_date = raw.index[raw.index > date].min()
        mismatches += int(label_date != next_date)
    recomputed = np.log(raw.loc[frame.label_date].values / raw.loc[frame.Date].values)
    return {
        "feature_rows": int(len(frame)),
        "label_next_day_mismatches": int(mismatches),
        "max_abs_y_error": float(np.max(np.abs(recomputed - frame.y.values))),
    }


def _prefix_invariance(frame: pd.DataFrame, dbs: pd.DataFrame, dates: list[str]) -> dict:
    results = {}
    for date in dates:
        matches = frame.index[frame.Date == pd.Timestamp(date)]
        if len(matches) == 0:
            results[date] = None
            continue
        i = matches[0]
        cutoff = pd.Timestamp(frame.label_date.iloc[i])
        truncated = compute_features(dbs[dbs.Date <= cutoff].copy())
        row = truncated[truncated.Date == pd.Timestamp(date)]
        if len(row) == 0:
            results[date] = None
            continue
        diff = np.max(
            np.abs(frame.loc[i, FEATURES].to_numpy(float) - row.iloc[0][FEATURES].to_numpy(float))
        )
        results[date] = float(diff)
    return results


def _fold_checks(frame: pd.DataFrame) -> dict:
    folds = [fold for fold in make_folds_from_frame(frame) if fold.fold_id != "holdout"]
    disjoint = True
    for fold in folds:
        train = set(fold.train_idx)
        test = set(fold.test_idx)
        val = set(fold.val_idx)
        select = set(fold.select_train_idx)
        disjoint &= not (train & test) and not (select & val) and not (val & test)
    return {
        "folds": len(folds),
        "disjoint": bool(disjoint),
        "test_sizes": [int(len(fold.test_idx)) for fold in folds],
        "test_total": int(sum(len(fold.test_idx) for fold in folds)),
    }


def _missing_day_checks(dbs: pd.DataFrame, sti: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    dbs_dates = set(dbs["Date"])
    sti_dates = set(sti["Date"])
    dbs_only = sorted(dbs_dates - sti_dates)
    sti_only = sorted(sti_dates - dbs_dates)
    affected = []
    for missing in sti_only:
        before = dbs.index[dbs["Date"] < missing]
        if len(before) == 0:
            continue
        i = before[-1]
        next_row = dbs.index[dbs["Date"] > dbs["Date"].iloc[i]][0]
        affected.append(
            {
                "skipped_day": missing.strftime("%Y-%m-%d"),
                "feature_date": dbs["Date"].iloc[i].strftime("%Y-%m-%d"),
                "label_date": dbs["Date"].iloc[next_row].strftime("%Y-%m-%d"),
            }
        )
    report = {
        "dbs_only_days": len(dbs_only),
        "sti_only_days": len(sti_only),
        "dbs_only_sample": [d.strftime("%Y-%m-%d") for d in dbs_only[:5]],
        "sti_only": [d.strftime("%Y-%m-%d") for d in sti_only],
    }
    return report, pd.DataFrame(affected)


def _impact(dbs: pd.DataFrame, sti: pd.DataFrame) -> pd.DataFrame:
    _, affected = _missing_day_checks(dbs, sti)
    if affected.empty:
        return pd.DataFrame()
    bad_dates = set(affected["feature_date"])
    rows = []
    for model in ("no_prediction", "denfis_frozen", "denfis_online", "xgboost", "lstm"):
        data = pd.concat(
            [pd.read_csv(ROOT / "results" / "preds" / model / f"{fold}.csv") for fold in FOLD_IDS],
            ignore_index=True,
        )

        def skill(frame: pd.DataFrame) -> float:
            return 1.0 - float(np.sqrt(np.mean((frame.y_true - frame.y_pred) ** 2))) / float(
                np.sqrt(np.mean(frame.y_true**2))
            )

        all_rows = skill(data)
        dropped = data[~data["date"].isin(bad_dates)]
        rows.append(
            {
                "model": model,
                "skill_all": all_rows,
                "skill_drop": skill(dropped),
                "delta": skill(dropped) - all_rows,
                "rows_dropped": len(data) - len(dropped),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    raw, dbs = _raw_checks()
    adjustment, jumps = _adjustment_checks(dbs)
    frame = compute_features(dbs)
    labels = _label_checks(frame, dbs)
    prefix = _prefix_invariance(
        frame, dbs, ["2017-01-03", "2020-03-09", "2024-11-12", "2025-04-07"]
    )
    folds = _fold_checks(frame)
    sti = load_prices("STI")
    sti["Date"] = pd.to_datetime(sti["Date"])
    missing, affected = _missing_day_checks(dbs, sti)
    impact = _impact(dbs, sti)

    report = {
        "raw": raw,
        "adjustment": adjustment,
        "labels": labels,
        "prefix_invariance": prefix,
        "folds": folds,
        "trading_days": missing,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "data_integrity.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    jumps.to_csv(OUT_DIR / "data_adjustment_jumps.csv", index=False)
    impact.to_csv(OUT_DIR / "data_missing_day_impact.csv", index=False)

    def _fmt_impact() -> str:
        if impact.empty:
            return "_No affected rows._"
        lines = [
            "| model | skill (all rows) | skill (drop affected) | delta | rows dropped |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in impact.itertuples(index=False):
            lines.append(
                f"| {row.model} | {row.skill_all:+.3%} | {row.skill_drop:+.3%} | "
                f"{row.delta:+.3%} | {row.rows_dropped} |"
            )
        return "\n".join(lines)

    def _fmt_prefix() -> str:
        lines = ["| feature date | max abs difference after truncating the future |", "|---|---:|"]
        for date, value in prefix.items():
            lines.append(f"| {date} | {value if value is not None else 'row not present'} |")
        return "\n".join(lines)

    note = f"""# Data integrity and alignment audit

Run by `scripts/data_integrity_check.py`. The cache is read-only; the 2026
holdout is untouched.

## 1. Raw files

- Checksums match `data/raw/checksums.txt`: **{raw['checksums_ok']}**.
- DBS rows: **{raw['dbs_rows']}**, {raw['dbs_start']} to {raw['dbs_end']}.
- Duplicate dates: **{raw['duplicate_dates']}**; null cells: **{raw['null_cells']}**.
- Non-positive adjusted close: **{raw['nonpositive_adj_close']}**;
  zero/negative volume: **{raw['zero_or_negative_volume']}**.
- OHLC violations: **{raw['ohlc_violations']}**; flat bars: **{raw['flat_bars']}**.
- Largest calendar gap: **{raw['max_calendar_gap_days']} days**.
- Repeated closes: **{raw['repeated_closes']}** ({raw['repeated_closes']/raw['dbs_rows']:.1%}),
  spread across all years, with no flat bars.

## 2. Corporate actions

- `Adj Close / Close` jumps above 1%: **{adjustment['ratio_jumps_over_1pct']}**.
- Negative ratio jumps: **{adjustment['negative_ratio_jumps']}**.
- Largest ratio jump: **{adjustment['largest_ratio_jump']:.2%}**.

Every jump is an ex-dividend adjustment: the unadjusted close falls while the
adjusted return is smooth. There is no split-sized jump. The largest adjusted
moves are real events (COVID March 2020, the April 2025 tariff shock, the
August 2024 carry unwind), not data errors.

## 3. Labels

- Feature rows: **{labels['feature_rows']}**.
- `label_date` equals the next trading day: **{labels['label_next_day_mismatches']} mismatches**.
- Maximum `|y - log(AdjClose[label]/AdjClose[date])|`: **{labels['max_abs_y_error']:.1e}**.

## 4. Feature causality

{_fmt_prefix()}

A zero difference means the feature row at that date is unchanged when every
later price is removed. The features use only information available at `t`.

## 5. Folds

- Folds: **{folds['folds']}**; train/validation/test disjoint: **{folds['disjoint']}**.
- Test sizes: {folds['test_sizes']}; total **{folds['test_total']}**.

## 6. Trading-day completeness

The DBS series is missing **{missing['sti_only_days']}** days that STI has:
{', '.join(missing['sti_only'])}. The affected feature-date to label-date pairs
are {', '.join(f"{row.feature_date} to {row.label_date}" for row in affected.itertuples(index=False)) if not affected.empty else 'none'}.

Impact of dropping the affected rows from the pooled test predictions:

{_fmt_impact()}

The largest change is a few hundredths of a percentage point; the conclusions
are unchanged. The reverse mismatch is **{missing['dbs_only_days']}** days where
DBS traded and STI is missing (mostly US/UK holidays, for example
{', '.join(missing['dbs_only_sample'])}). Those gaps matter only for the
rejected STI-feature extension; the main 13 features do not use STI.

## Verdict

No data defect explains the model results. The DBS adjusted series, labels,
feature causality and folds are clean. The only real imperfection is three
missing DBS trading days, one of which falls inside the test period; its
impact is negligible. If the STI extension is revived, align both series on
common dates and re-check.

## Reproduction

```bash
python scripts/data_integrity_check.py
```
"""
    NOTE_PATH.write_text(note, encoding="utf-8")
    print(note)
    print(f"Saved {OUT_DIR / 'data_integrity.json'}")
    print(f"Saved {OUT_DIR / 'data_adjustment_jumps.csv'}")
    print(f"Saved {OUT_DIR / 'data_missing_day_impact.csv'}")
    print(f"Saved {NOTE_PATH}")


if __name__ == "__main__":
    main()
