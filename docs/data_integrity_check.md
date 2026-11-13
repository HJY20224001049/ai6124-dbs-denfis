# Data integrity and alignment audit

Run by `scripts/data_integrity_check.py`. The cache is read-only; the 2026
holdout is untouched.

## 1. Raw files

- Checksums match `data/raw/checksums.txt`: **True**.
- DBS rows: **4219**, 2010-01-04 to 2026-10-30.
- Duplicate dates: **0**; null cells: **0**.
- Non-positive adjusted close: **0**;
  zero/negative volume: **0**.
- OHLC violations: **0**; flat bars: **0**.
- Largest calendar gap: **5 days**.
- Repeated closes: **126** (3.0%),
  spread across all years, with no flat bars.

## 2. Corporate actions

- `Adj Close / Close` jumps above 1%: **43**.
- Negative ratio jumps: **0**.
- Largest ratio jump: **3.73%**.

Every jump is an ex-dividend adjustment: the unadjusted close falls while the
adjusted return is smooth. There is no split-sized jump. The largest adjusted
moves are real events (COVID March 2020, the April 2025 tariff shock, the
August 2024 carry unwind), not data errors.

## 3. Labels

- Feature rows: **4158**.
- `label_date` equals the next trading day: **0 mismatches**.
- Maximum `|y - log(AdjClose[label]/AdjClose[date])|`: **0.0e+00**.

## 4. Feature causality

| feature date | max abs difference after truncating the future |
|---|---:|
| 2017-01-03 | 0.0 |
| 2020-03-09 | 0.0 |
| 2024-11-12 | 0.0 |
| 2025-04-07 | 0.0 |

A zero difference means the feature row at that date is unchanged when every
later price is removed. The features use only information available at `t`.

## 5. Folds

- Folds: **9**; train/validation/test disjoint: **True**.
- Test sizes: [250, 251, 250, 251, 253, 250, 249, 252, 252]; total **2258**.

## 6. Trading-day completeness

The DBS series is missing **3** days that STI has:
2010-12-31, 2012-04-02, 2020-03-18. The affected feature-date to label-date pairs
are 2010-12-30 to 2011-01-03, 2012-03-30 to 2012-04-03, 2020-03-17 to 2020-03-19.

Impact of dropping the affected rows from the pooled test predictions:

| model | skill (all rows) | skill (drop affected) | delta | rows dropped |
|---|---:|---:|---:|---:|
| no_prediction | +0.000% | +0.000% | +0.000% | 1 |
| denfis_frozen | -0.759% | -0.751% | +0.008% | 1 |
| denfis_online | -1.756% | -1.754% | +0.002% | 1 |
| xgboost | -0.283% | -0.260% | +0.022% | 1 |
| lstm | -0.569% | -0.529% | +0.040% | 1 |

The largest change is a few hundredths of a percentage point; the conclusions
are unchanged. The reverse mismatch is **12** days where
DBS traded and STI is missing (mostly US/UK holidays, for example
2010-08-02, 2011-02-21, 2011-05-30, 2011-07-04, 2011-11-24). Those gaps matter only for the
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
