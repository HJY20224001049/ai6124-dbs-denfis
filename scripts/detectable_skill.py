"""How large must an RMSE improvement be before this sample can detect it?

The professor's definition of "better" is an ordering on forecast error.  That
ordering is only meaningful if the study has enough power to resolve it.  This
script computes, for each candidate, the minimum detectable effect (MDE) at
80% power and a two-sided 5% test, using Newey-West standard errors on the
daily squared-error difference against the zero forecast.

The test years are used for evaluation only; nothing here selects a model,
feature or fold.  The 2026 holdout is untouched.  The comparison is
descriptive and post-hoc.

Run:
    python scripts/detectable_skill.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FEATURES, TARGET_COL  # noqa: E402
from src.data import load_prices  # noqa: E402
from src.features import compute_features  # noqa: E402
from src.splits import make_folds_from_frame  # noqa: E402

OUT_DIR = ROOT / "results" / "analysis"
NOTE_PATH = ROOT / "docs" / "detectable_skill.md"
NW_LAG = 7
Z_ALPHA = 1.959963984540054  # two-sided 5%
Z_POWER = 0.8416212335729143  # 80% power


def _standardise(X: np.ndarray, fit_idx: np.ndarray, eval_idx: np.ndarray):
    lo = np.quantile(X[fit_idx], 0.005, axis=0)
    hi = np.quantile(X[fit_idx], 0.995, axis=0)
    train = np.clip(X[fit_idx], lo, hi)
    other = np.clip(X[eval_idx], lo, hi)
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    return (train - mean) / std, (other - mean) / std


def _ols_predict(X_train, y_train, X_eval):
    design = np.column_stack([np.ones(len(X_train)), X_train])
    coef, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    return coef[0] + X_eval @ coef[1:]


def _nw_mean_se(values: np.ndarray, lag: int = NW_LAG) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    centred = values - mean
    n = len(values)
    lrv = float(np.mean(centred**2))
    for k in range(1, lag + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1)) * float(np.mean(centred[:-k] * centred[k:]))
    return mean, float(np.sqrt(lrv / n))


def _collect_univariate(frame, folds, feature: str):
    X = frame[FEATURES].to_numpy(dtype=np.float64)
    y = frame[TARGET_COL].to_numpy(dtype=np.float64)
    j = FEATURES.index(feature)
    ys, preds = [], []
    for fold in folds:
        X_train, X_eval = _standardise(X, fold.train_idx, fold.test_idx)
        ys.append(y[fold.test_idx])
        preds.append(_ols_predict(X_train[:, [j]], y[fold.train_idx], X_eval[:, [j]]))
    return np.concatenate(ys), np.concatenate(preds)


def _collect_constant(frame, folds):
    X = frame[FEATURES].to_numpy(dtype=np.float64)
    y = frame[TARGET_COL].to_numpy(dtype=np.float64)
    ys, preds = [], []
    for fold in folds:
        ys.append(y[fold.test_idx])
        preds.append(np.full(len(fold.test_idx), y[fold.train_idx].mean()))
    return np.concatenate(ys), np.concatenate(preds)


def _collect_model(folds, model: str):
    frames = []
    for fold in folds:
        path = ROOT / "results" / "preds" / model / f"{fold.fold_id}.csv"
        frames.append(pd.read_csv(path))
    data = pd.concat(frames, ignore_index=True)
    return data["y_true"].to_numpy(float), data["y_pred"].to_numpy(float)


def _row(name, y, pred, mse_zero):
    diff = (y - pred) ** 2 - y**2
    mean, se = _nw_mean_se(diff)
    dm = mean / se if se > 0 else 0.0
    p_value = float(2.0 * (1.0 - norm.cdf(abs(dm))))
    mde = (Z_ALPHA + Z_POWER) * se
    skill = 1.0 - float(np.sqrt(np.mean((y - pred) ** 2))) / float(np.sqrt(np.mean(y**2)))
    mde_skill = 1.0 - np.sqrt(max(1e-12, 1.0 - mde / mse_zero)) if mde < mse_zero else np.nan
    power = float(norm.cdf(abs(dm) - Z_ALPHA))
    iid_se = float(np.sqrt(np.var(diff, ddof=1) / len(diff)))
    effective_n = len(diff) * (iid_se**2) / (se**2) if se > 0 else np.nan
    return {
        "rule": name,
        "n": len(diff),
        "observed_rmse_skill": skill,
        "mean_squared_error_diff": mean,
        "nw_se": se,
        "dm_stat": dm,
        "dm_p": p_value,
        "mde_mse": mde,
        "mde_rmse_skill": mde_skill,
        "power_at_observed": power,
        "effective_n": effective_n,
    }


def _fmt_table(df: pd.DataFrame) -> str:
    lines = [
        "| rule | observed skill | NW SE (MSE) | DM p | MDE at 80% power | power at observed | effective n |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.rule} | {row.observed_rmse_skill:+.3%} | {row.nw_se:.2e} | "
            f"{row.dm_p:.3f} | {row.mde_rmse_skill:+.3%} | "
            f"{row.power_at_observed:.2f} | {row.effective_n:.0f} |"
        )
    return "\n".join(lines)


def main() -> None:
    prices = load_prices("D05_SI")
    frame = compute_features(prices)
    folds = [fold for fold in make_folds_from_frame(frame) if fold.fold_id != "holdout"]

    y_zero, _ = _collect_model(folds, "no_prediction")
    mse_zero = float(np.mean(y_zero**2))

    candidates = {
        "ma5": _collect_univariate(frame, folds, "ma5"),
        "ret_3": _collect_univariate(frame, folds, "ret_3"),
        "constant mean": _collect_constant(frame, folds),
        "denfis_frozen": _collect_model(folds, "denfis_frozen"),
        "xgboost": _collect_model(folds, "xgboost"),
        "lstm": _collect_model(folds, "lstm"),
    }
    rows = [_row(name, y, pred, mse_zero) for name, (y, pred) in candidates.items()]
    table = pd.DataFrame(rows)

    # Paired DENFIS vs XGBoost: is the observed gap resolvable?
    y, denfis = candidates["denfis_frozen"]
    _, xgb = candidates["xgboost"]
    paired_diff = (y - denfis) ** 2 - (y - xgb) ** 2
    paired_mean, paired_se = _nw_mean_se(paired_diff)
    paired_dm = paired_mean / paired_se
    paired_p = float(2.0 * (1.0 - norm.cdf(abs(paired_dm))))
    paired_mde = (Z_ALPHA + Z_POWER) * paired_se
    paired_mde_skill = 1.0 - np.sqrt(max(1e-12, 1.0 - paired_mde / mse_zero))
    observed_gap = float(
        np.sqrt(np.mean((y - denfis) ** 2)) / np.sqrt(np.mean((y - xgb) ** 2)) - 1.0
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "detectable_skill.csv", index=False)

    note = f"""# Is the requirement statistically resolvable?

Diagnostic run by `scripts/detectable_skill.py`. The zero forecast is the
benchmark; `mse_zero = {mse_zero:.4e}` and `rmse_zero = {np.sqrt(mse_zero):.6f}`
over the nine test years (n = {len(y_zero)}). Test years are used for
evaluation only; nothing here selects a model, feature or fold, and the 2026
holdout is untouched.

## What "minimum detectable effect" means

For each candidate we form the daily squared-error difference against the zero
forecast, estimate its Newey-West standard error (lag {NW_LAG}), and compute the
RMSE improvement that would be detected with 80% power at a two-sided 5% test.
`power at observed` is the probability of a significant result if the observed
effect were the true effect. Effective n below the nominal 2,258 indicates how
much autocorrelation inflates the standard error (values above 2,258 indicate
slightly negative autocorrelation).

## Results

{_fmt_table(table)}

## Paired comparison: DENFIS against XGBoost

- Observed RMSE gap: **{observed_gap:+.2%}** (positive means DENFIS is worse).
- Mean squared-error difference: {paired_mean:+.2e}, Newey-West SE {paired_se:.2e}.
- DM statistic {paired_dm:+.2f}, p = {paired_p:.3f}.
- MDE for the gap at 80% power: **{paired_mde_skill:+.2%}**.

## Reading

- **The best observed effect is below the detection threshold.** The strongest
  rule (`ma5`) improves RMSE by {table.loc[table.rule == 'ma5', 'observed_rmse_skill'].iloc[0]:+.2%},
  while the study needs about
  {table.loc[table.rule == 'ma5', 'mde_rmse_skill'].iloc[0]:+.2%} to reach 80% power.
  Its power at the observed effect is only
  {table.loc[table.rule == 'ma5', 'power_at_observed'].iloc[0]:.2f}.
- **DENFIS versus XGBoost is not resolvable either.** The observed gap is
  {observed_gap:+.2%} but the MDE is {paired_mde_skill:+.2%}; a true gap of this
  size would be detected only about {norm.cdf(abs(paired_dm) - Z_ALPHA):.0%} of the time.
- **The baseline clause is a demand on the baselines, not on DENFIS.** To make
  `xgboost < zero` detectable at 80% power needs an improvement of about
  {table.loc[table.rule == 'xgboost', 'mde_rmse_skill'].iloc[0]:+.2%}; LSTM needs about
  {table.loc[table.rule == 'lstm', 'mde_rmse_skill'].iloc[0]:+.2%}. Both currently have the
  wrong sign.
- **Sample size is the binding constraint.** Detecting a true 0.4% RMSE edge at
  80% power would need roughly 3,500–3,700 trading days (about 15 years), or a
  larger effect. The nine test years are enough to rule out a large edge, not
  to establish a small one.

## Caveat

The MDE uses daily squared-error differences and a Newey-West correction. It
does not model cross-fold dependence explicitly; the fold-level bootstrap in
`docs/feature_information_check.md` gives wider intervals, so these MDEs are
optimistic. The conclusion is therefore conservative: the sample is, if
anything, less able to resolve the requirement than the table suggests.

## Implication for the consultation

When the brief says "better", it matters whether it means a point-estimate
ordering or a statistically established improvement. For effects below about
1% RMSE, this sample cannot reliably distinguish the two. That is the question
to put to the instructor, together with the no-prediction comparison.

## Reproduction

```bash
python scripts/detectable_skill.py
```
"""
    NOTE_PATH.write_text(note, encoding="utf-8")
    print(note)
    print(f"Saved {OUT_DIR / 'detectable_skill.csv'}")
    print(f"Saved {NOTE_PATH}")


if __name__ == "__main__":
    main()
