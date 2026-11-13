"""Diagnostic: do the 13 frozen features carry usable next-day information?

Two evaluation windows are reported separately:

* ``validation``: fit on the selection-training rows (labels through Y-2),
  score on the validation year Y-1.  This is the protocol's selection window.
* ``test``: fit on the final training rows (labels through Y-1), score on the
  test year Y.  Test years are used for evaluation only; nothing here selects
  a feature, a hyper-parameter or a fold.  The 2026 holdout is untouched.

For every feature we fit a two-parameter OLS rule (intercept plus that one
feature) and compare it with the zero forecast.  We also report the full
13-feature OLS, a fixed-alpha ridge, and a constant-mean forecast.  The
stored model predictions are added to the test table for comparison.

Because the nine folds are the independent units, uncertainty is quantified
by resampling folds (not days) and by a fold-level sign test.  A daily
Diebold-Mariano statistic is shown for reference only; pooling days across
nine years overstates its significance.

Run:
    python scripts/feature_information_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FEATURES, TARGET_COL  # noqa: E402
from src.data import load_prices  # noqa: E402
from src.features import compute_features  # noqa: E402
from src.splits import make_folds_from_frame  # noqa: E402

OUT_DIR = ROOT / "results" / "analysis"
NOTE_PATH = ROOT / "docs" / "feature_information_check.md"
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 123
DM_LAG = 7
RIDGE_ALPHA = 1.0


def _standardise(
    X: np.ndarray, fit_idx: np.ndarray, eval_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Winsorise and z-score using the fit window only."""
    lo = np.quantile(X[fit_idx], 0.005, axis=0)
    hi = np.quantile(X[fit_idx], 0.995, axis=0)
    train = np.clip(X[fit_idx], lo, hi)
    other = np.clip(X[eval_idx], lo, hi)
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    return (train - mean) / std, (other - mean) / std


def _ols_predict(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(X_train)), X_train])
    coef, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    return coef[0] + X_eval @ coef[1:]


def _pooled(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    y = np.concatenate([pair[0] for pair in pairs])
    pred = np.concatenate([pair[1] for pair in pairs])
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    rmse_zero = float(np.sqrt(np.mean(y**2)))
    skill = 1.0 - rmse / rmse_zero
    ic = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 0 else 0.0
    return skill, ic


def _fold_skill(pair: tuple[np.ndarray, np.ndarray]) -> float:
    y, pred = pair
    return 1.0 - float(np.sqrt(np.mean((y - pred) ** 2))) / float(np.sqrt(np.mean(y**2)))


def _bootstrap_ci(pairs: list[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator) -> tuple[float, float]:
    n = len(pairs)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for i in range(BOOTSTRAP_DRAWS):
        picks = rng.integers(0, n, n)
        draws[i] = _pooled([pairs[j] for j in picks])[0]
    return tuple(float(v) for v in np.percentile(draws, [2.5, 97.5]))


def _leave_one_out(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    skills = [_pooled([p for j, p in enumerate(pairs) if j != i])[0] for i in range(len(pairs))]
    return float(min(skills)), float(max(skills))


def _dm_vs_zero(pair: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
    y, pred = pair
    diff = (y - pred) ** 2 - y**2
    mean = float(diff.mean())
    centred = diff - mean
    n = len(diff)
    lrv = float(np.mean(centred**2))
    for lag in range(1, DM_LAG + 1):
        lrv += 2.0 * (1.0 - lag / (DM_LAG + 1)) * float(np.mean(centred[:-lag] * centred[lag:]))
    se = np.sqrt(lrv / n)
    stat = mean / se if se > 0 else 0.0
    return float(stat), float(2.0 * (1.0 - norm.cdf(abs(stat))))


def _collect(
    frame: pd.DataFrame,
    folds,
    eval_set: str,
) -> tuple[dict[str, list[tuple[np.ndarray, np.ndarray]]], list[str]]:
    X = frame[FEATURES].to_numpy(dtype=np.float64)
    y = frame[TARGET_COL].to_numpy(dtype=np.float64)
    rules = {name: [] for name in FEATURES}
    rules.update({"ALL_LINEAR": [], "RIDGE": [], "CONST_MEAN": []})
    fold_ids = []

    for fold in folds:
        fit_idx = fold.select_train_idx if eval_set == "validation" else fold.train_idx
        eval_idx = fold.val_idx if eval_set == "validation" else fold.test_idx
        X_train, X_eval = _standardise(X, fit_idx, eval_idx)
        y_train, y_eval = y[fit_idx], y[eval_idx]
        fold_ids.append(fold.fold_id)

        for j, name in enumerate(FEATURES):
            rules[name].append((y_eval, _ols_predict(X_train[:, [j]], y_train, X_eval[:, [j]])))
        rules["ALL_LINEAR"].append((y_eval, _ols_predict(X_train, y_train, X_eval)))
        ridge = Ridge(alpha=RIDGE_ALPHA).fit(X_train, y_train)
        rules["RIDGE"].append((y_eval, ridge.predict(X_eval)))
        rules["CONST_MEAN"].append((y_eval, np.full(len(y_eval), y_train.mean())))

    return rules, fold_ids


def _add_stored_models(
    rules: dict[str, list[tuple[np.ndarray, np.ndarray]]], fold_ids: list[str]
) -> None:
    for model in ("denfis_frozen", "denfis_online", "xgboost", "lstm", "no_prediction"):
        pairs = []
        for fold_id in fold_ids:
            path = ROOT / "results" / "preds" / model / f"{fold_id}.csv"
            if not path.exists():
                break
            pred = pd.read_csv(path)
            pairs.append((pred["y_true"].to_numpy(float), pred["y_pred"].to_numpy(float)))
        if len(pairs) == len(fold_ids):
            rules[model] = pairs


def _top_k_table(
    frame: pd.DataFrame,
    folds,
    ranking: list[str],
) -> pd.DataFrame:
    """Fit OLS on the top-k validation-ranked features and score each test year.

    Selection uses validation skill only.  The test columns are descriptive:
    they show whether a small combination preserves the univariate signal, not
    a new model chosen on test performance.
    """
    X = frame[FEATURES].to_numpy(dtype=np.float64)
    y = frame[TARGET_COL].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for k in range(1, 9):
        selected = ranking[:k]
        columns = [FEATURES.index(name) for name in selected]
        pairs = []
        for fold in folds:
            X_train, X_eval = _standardise(X, fold.train_idx, fold.test_idx)
            pairs.append(
                (
                    y[fold.test_idx],
                    _ols_predict(X_train[:, columns], y[fold.train_idx], X_eval[:, columns]),
                )
            )
        skill, ic = _pooled(pairs)
        ci_low, ci_high = _bootstrap_ci(pairs, rng)
        fold_skills = [_fold_skill(pair) for pair in pairs]
        wins = sum(value > 0 for value in fold_skills)
        rows.append(
            {
                "k": k,
                "features": ", ".join(selected),
                "pooled_rmse_skill": skill,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "folds_positive": wins,
                "sign_test_p": float(binomtest(wins, len(pairs), 0.5).pvalue),
                "ic": ic,
            }
        )
    return pd.DataFrame(rows)


def _fmt_top_k(df: pd.DataFrame) -> str:
    lines = [
        "| k | features | pooled skill | fold-bootstrap 95% CI | folds > 0 | sign p | IC |",
        "|---:|---|---:|---|---:|---:|---:|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.k} | {row.features} | {row.pooled_rmse_skill:+.3%} | "
            f"[{row.ci_low:+.3%}, {row.ci_high:+.3%}] | {row.folds_positive}/9 | "
            f"{row.sign_test_p:.3f} | {row.ic:+.4f} |"
        )
    return "\n".join(lines)


def _summarise(
    rules: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    eval_set: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for name, pairs in rules.items():
        skill, ic = _pooled(pairs)
        ci_low, ci_high = _bootstrap_ci(pairs, rng)
        fold_skills = [_fold_skill(pair) for pair in pairs]
        wins = sum(value > 0 for value in fold_skills)
        loo_low, loo_high = _leave_one_out(pairs)
        dm_stat, dm_p = _dm_vs_zero((np.concatenate([p[0] for p in pairs]),
                                     np.concatenate([p[1] for p in pairs])))
        rows.append(
            {
                "eval_set": eval_set,
                "rule": name,
                "pooled_rmse_skill": skill,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "folds_positive": wins,
                "sign_test_p": float(binomtest(wins, len(pairs), 0.5).pvalue),
                "leave_one_out_min": loo_low,
                "leave_one_out_max": loo_high,
                "ic": ic,
                "dm_vs_zero": dm_stat,
                "dm_p": dm_p,
            }
        )
    return pd.DataFrame(rows)


def _fmt_table(df: pd.DataFrame) -> str:
    lines = [
        "| rule | pooled skill | fold-bootstrap 95% CI | folds > 0 | sign p | leave-one-out range | IC |",
        "|---|---:|---|---:|---:|---|---:|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.rule} | {row.pooled_rmse_skill:+.3%} | "
            f"[{row.ci_low:+.3%}, {row.ci_high:+.3%}] | "
            f"{row.folds_positive}/9 | {row.sign_test_p:.3f} | "
            f"[{row.leave_one_out_min:+.3%}, {row.leave_one_out_max:+.3%}] | "
            f"{row.ic:+.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    prices = load_prices("D05_SI")
    frame = compute_features(prices)
    folds = [fold for fold in make_folds_from_frame(frame) if fold.fold_id != "holdout"]

    validation_rules, fold_ids = _collect(frame, folds, "validation")
    test_rules, _ = _collect(frame, folds, "test")
    _add_stored_models(test_rules, fold_ids)

    validation = _summarise(validation_rules, "validation")
    test = _summarise(test_rules, "test")
    ranking = list(
        validation.loc[validation["rule"].isin(FEATURES)]
        .sort_values("pooled_rmse_skill", ascending=False)["rule"]
    )
    top_k = _top_k_table(frame, folds, ranking)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    validation.to_csv(OUT_DIR / "feature_information_validation.csv", index=False)
    test.to_csv(OUT_DIR / "feature_information_test.csv", index=False)
    top_k.to_csv(OUT_DIR / "feature_information_top_k.csv", index=False)

    order = (
        ["ma5", "boll_b", "ret_0", "ret_1", "ret_2", "ret_3", "ret_4",
         "ma20", "ma60", "rsi14", "macd", "rv20", "vol_z",
         "CONST_MEAN", "ALL_LINEAR", "RIDGE"]
        + ["no_prediction", "denfis_frozen", "denfis_online", "xgboost", "lstm"]
    )
    validation = validation.set_index("rule").reindex(
        [name for name in order if name in validation["rule"].values]
    ).reset_index()
    test = test.set_index("rule").reindex(
        [name for name in order if name in test["rule"].values]
    ).reset_index()

    note = f"""# Do the 13 features carry next-day information?

Diagnostic run by `scripts/feature_information_check.py`. Nothing here selects
a feature, hyper-parameter or fold on test performance, and the 2026 holdout
is untouched. The test table evaluates rules fitted on the final training
window; it is descriptive, not a new model selection.

## Validation years (fit on labels through Y-2, score Y-1)

{_fmt_table(validation)}

## Test years (fit on labels through Y-1, score Y; evaluation only)

{_fmt_table(test)}

`no_prediction` is the benchmark row; its sign test is degenerate (all zeros)
and should be ignored.

## Simple combinations (features ranked on validation, evaluated on test)

{_fmt_top_k(top_k)}

## Reading

- **There is weak univariate signal, but it is not fold-robust.** Several
  single-feature rules have fold-bootstrap intervals that exclude zero on the
  test years (`ret_1` to `ret_4`, `rsi14`, `rv20`, `ma60`, and the constant
  drift). Their point estimates are tiny (+0.09% to +0.19%), most win only
  6 of 9 folds, and the fold-level sign tests are not significant. `ma5` has
  the largest point estimate (+0.39%) but its interval includes zero.
- **The interval and the sign test disagree by construction.** The bootstrap
  is magnitude-weighted over days, while the sign test counts folds. With
  nine noisy years, a few large positive folds can pull the pooled interval
  away from zero without a consistent fold-level edge.
- **The joint linear model destroys the univariate signal.** The 13-feature
  OLS and the fixed-alpha ridge are negative on both windows even though many
  single-feature rules are positive. With 13 correlated inputs and a
  signal-to-noise ratio this low, estimation variance dominates.
- **Small combinations do not rescue it.** Top-k models selected on validation
  stay positive on test for k = 1 to 8, but every interval includes zero and
  no model wins more than 6 of 9 folds. The best simple rule is still not a
  statistically established edge.
- **The stored models fail to exploit what little signal exists.** DENFIS,
  XGBoost and LSTM are negative on test; DENFIS and LSTM are negative even in
  every leave-one-fold-out variant. This is consistent with over-fitting a
  weak signal, not with a coding error; the synthetic-signal check already
  shows the same pipeline learns a planted signal.

## Caveat for any follow-up

The test years have now been inspected in this diagnostic. A model designed
from these results cannot be sold as a clean test-set result. The defensible
route is to design it on validation only, freeze it, and evaluate once on the
untouched 2026 holdout.

## Reproduction

```bash
python scripts/feature_information_check.py
```
"""
    NOTE_PATH.write_text(note, encoding="utf-8")
    print(note)
    print(f"Saved {OUT_DIR / 'feature_information_validation.csv'}")
    print(f"Saved {OUT_DIR / 'feature_information_test.csv'}")
    print(f"Saved {OUT_DIR / 'feature_information_top_k.csv'}")
    print(f"Saved {NOTE_PATH}")


if __name__ == "__main__":
    main()
