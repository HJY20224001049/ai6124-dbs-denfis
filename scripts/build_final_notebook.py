"""Build and execute the final AI6124 Group 18 submission notebook.

The notebook is assembled from the repository's frozen artifacts and source
modules, then executed on a fresh kernel so the committed notebook carries real
outputs. Sections follow PROTOCOL.md section 10.1.

No experiment artifact is rewritten: the live-reproduction section calls
``run_fold`` directly, which returns a prediction frame without saving it.

Usage (from the repository root):

    python scripts/build_final_notebook.py            # build and execute
    python scripts/build_final_notebook.py --no-exec  # build only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "writing/AI6124_Project_Group18_HuJunyu_LinJingzhao.ipynb"

STUDENT_IDS = {"LIN Jingzhao": "G2605587L", "HU Junyu": "G2603713B"}


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


def listing(relative: str, lang: str = "python") -> str:
    """Fenced code block for the appendix listings."""
    text = (ROOT / relative).read_text(encoding="utf-8")
    return f"**`{relative}`**\n\n```{lang}\n{text.rstrip()}\n```"


# ---------------------------------------------------------------------------
# Section 0 — title page
# ---------------------------------------------------------------------------
def section0() -> list:
    return [md(f"""
# Neuro-fuzzy local modelling of next-day DBS returns

**AI6124 Neuro-Fuzzy Systems — Group 18**

DENFIS (evolving clustering → first-order Takagi–Sugeno rules → weighted recursive
least squares) against XGBoost and LSTM baselines, on daily bars of DBS Group
Holdings (`D05.SI`).

| Member | Student ID | Responsibility |
|---|---|---|
| LIN Jingzhao | {STUDENT_IDS['LIN Jingzhao']} | Method line: DENFIS implementation, evolving-rule and transfer analysis, method write-up |
| HU Junyu | {STUDENT_IDS['HU Junyu']} | Data and evaluation line: data pipeline, 13 causal features, walk-forward folds, baselines, backtest and statistical tests |

### How to run

* Open this notebook from the repository root (running it from `writing/` also
  works — the first cell locates the repository root).
* Choose **Kernel → Restart & Run All**. No network access is used: every input
  is the cached CSV in `data/raw/`, verified against `data/checksums.txt`.
* A full run takes roughly one minute. The nine-fold tables, figures and
  statistical tests are read from `results/`, which the repository's scripts
  produced; one fold is re-trained end to end and checked against those saved
  predictions in Section 7.5.
* Dependencies are pinned in `requirements.txt`. The notebook was verified on
  Python 3.9 and 3.11.

### Status and scope of the claims

This notebook reports **nine chronological test years, 2017–2025**. The 2026
holdout is reported separately in Section 7.6 and is not used anywhere for
model selection. Results are preliminary until the scheduled freeze recorded in
`docs/freeze.md`; the git commit that produced the artifacts is printed in
Section 1 so that any figure can be traced back to a specific run.

The headline finding is deliberately unflattering: none of the three models
forecasts next-day returns well enough to beat buy-and-hold after costs, and no
pairwise difference in forecast accuracy is statistically significant. Section 8
explains what the fitted rules do show, and where the evidence stops.
""")]


# ---------------------------------------------------------------------------
# Section 1 — setup
# ---------------------------------------------------------------------------
def section1() -> list:
    return [
        md("""
## 1. Setup

Everything below depends on `config.py`, which is the project's single source of
truth for feature order, fold boundaries, search grids, seeds and cost
assumptions. Imports go through it rather than through copy-pasted constants, so
the notebook cannot silently drift from the scripts that produced `results/`.

The two checks that follow are not decoration. Cached market data is the only
input to every number in this report, so the first thing we do is confirm the
CSV files still hash to the values recorded when they were downloaded. A silent
edit to a price column would otherwise invalidate the whole comparison.
"""),
        code("""
from pathlib import Path
import hashlib
import os
import platform
import sys

# Locate the repository root whether the notebook is opened from the root or
# from writing/.
root = Path.cwd()
while not (root / "config.py").exists() and root != root.parent:
    root = root.parent
assert (root / "config.py").exists(), "Open this notebook inside the repository."
os.chdir(root)
sys.path.insert(0, str(root))

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 40)

from config import (                                                # noqa: E402
    ANNUALIZATION, COST_BPS, DATA_DIR, DATE_COL, DENFIS_GRID, FEATURES,
    FOLD_SPECS, GLOBAL_SEED, LSTM_CONFIGS, PROTOCOL_VERSION, RESULTS_DIR,
    TARGET_COL, TAU_GRID, XGB_GRID,
)

print("Repository root :", root)
print("Protocol version:", PROTOCOL_VERSION)
print("Global seed     :", GLOBAL_SEED)
"""),
        md("""
Package versions are recorded alongside the results so a reader can tell whether
a difference is a modelling change or an environment change.
"""),
        code("""
import scipy
import sklearn

VERSIONS = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": scipy.__version__,
    "scikit-learn": sklearn.__version__,
    "matplotlib": matplotlib.__version__,
}
for name in ("xgboost", "torch"):
    try:
        module = __import__(name)
        VERSIONS[name] = module.__version__
    except ImportError:                      # baselines are lazily imported
        VERSIONS[name] = "not installed"

pd.Series(VERSIONS, name="version").to_frame()
"""),
        md("""
The checksum check compares each cached file against `data/checksums.txt`. The
cell fails loudly on a mismatch rather than continuing with altered data.
"""),
        code("""
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

recorded = {}
for line in (DATA_DIR / "checksums.txt").read_text(encoding="utf-8").splitlines():
    parts = line.split()
    if len(parts) == 2:
        recorded[parts[1]] = parts[0]

rows = []
for name, expected in sorted(recorded.items()):
    actual = sha256(DATA_DIR / name)
    rows.append({"file": name, "recorded": expected[:16] + "…",
                 "actual": actual[:16] + "…", "match": actual == expected})

checksums = pd.DataFrame(rows)
assert checksums["match"].all(), "Cached data does not match data/checksums.txt"
checksums
"""),
    ]


# ---------------------------------------------------------------------------
# Section 2 — data
# ---------------------------------------------------------------------------
def section2() -> list:
    return [
        md("""
## 2. Data

The target is the next-day log return of DBS Group Holdings, a large Singapore
bank, over 2010-01-01 to 2026-10-31. Two decisions matter for everything that
follows:

* **Adjusted close is the price series.** Returns, moving averages, Bollinger
  bands and RSI are all computed on `Adj Close`, so dividends and splits do not
  appear as spurious returns. `Close` is kept for display only.
* **Rows are never forward-filled.** A missing bar is dropped. Filling it would
  create a zero return that the model would learn as real.

The `^STI` index is loaded as well: it is used only by the transfer experiment in
Section 8.4, never by the main comparison.
"""),
        code("""
from src.data import load_prices

dbs = load_prices("D05_SI")
sti = load_prices("STI")

summary_rows = []
for name, frame, is_index in [("D05.SI", dbs, False), ("^STI", sti, True)]:
    summary_rows.append({
        "series": name,
        "rows": len(frame),
        "first": str(frame[DATE_COL].min()),
        "last": str(frame[DATE_COL].max()),
        "missing OHLCV": int(frame[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
                             .isna().any(axis=1).sum()),
        "zero-volume rows": int((frame["Volume"] == 0).sum()),
        "duplicate dates": int(frame[DATE_COL].duplicated().sum()),
        "index series": is_index,
    })
pd.DataFrame(summary_rows).set_index("series")
"""),
        md("""
A quick look at the series before any modelling. The price panel carries the
usual information — a long uptrend with two sharp drawdowns — and the volume
panel shows the liquidity profile the `vol_z` feature will summarise.
"""),
        code("""
def show_png(relative: str, width: float = 10.0, aspect: float = 0.42):
    path = root / relative
    assert path.exists(), path
    figure, axis = plt.subplots(figsize=(width, width * aspect))
    axis.imshow(plt.imread(path))
    axis.axis("off")
    plt.show()

show_png("docs/figs/01_price_volume.png", 11.0, 0.38)
show_png("docs/figs/03_data_coverage.png", 11.0, 0.38)
"""),
        md("""
**Anomaly check.** The protocol requires every single-day move larger than 50%
to be inspected by hand, because a split or a bad print would otherwise enter the
training set as a genuine observation. This cell lists them with the surrounding
information needed to judge them.
"""),
        code("""
prices = dbs.set_index(DATE_COL)["Adj Close"].astype(float)
daily_return = np.log(prices / prices.shift(1))
extreme = daily_return[daily_return.abs() > 0.5]

print(f"{len(extreme)} trading days exceed |{0.5:.0%}| in absolute log return")
pd.DataFrame({
    "log return": extreme.map(lambda value: round(float(value), 4)),
    "adj close": prices.reindex(extreme.index).round(3),
}).rename_axis("date")
"""),
        md("""
The return series itself, and the anomaly panel produced during the data audit,
are reproduced here for the reader's convenience.
"""),
        code("""
show_png("docs/figs/02_returns.png", 11.0, 0.38)
show_png("docs/figs/04_return_anomalies.png", 11.0, 0.38)
"""),
    ]


# ---------------------------------------------------------------------------
# Section 3 — features
# ---------------------------------------------------------------------------
def section3() -> list:
    return [
        md("""
## 3. Features

Thirteen features, frozen in the protocol before any model was fitted. They fall
into three groups: recent return lags (the only direct memory of past returns),
trend and mean-reversion indicators built from moving averages, and a volatility /
volume block. The exact definitions live in `config.FEATURES` and
`src/features.py`; the table below is a reading aid, not a second specification.

| # | Name | Definition | Window |
|---|---|---|---|
| 1–5 | `ret_0` … `ret_4` | `ln(C_t / C_{t-1})`, lagged 0–4 days | 1 |
| 6–8 | `ma5`, `ma20`, `ma60` | `C_t / SMA_n(C)_t − 1` | 5, 20, 60 |
| 9 | `rsi14` | `(RSI14_t − 50) / 50`, Wilder smoothing | 14 |
| 10 | `macd` | `(EMA12 − EMA26 − EMA9(·)) / C_t`, `adjust=False` | 12/26/9 |
| 11 | `boll_b` | `(C_t − L_t) / (U_t − L_t)`, `U/L = SMA20 ± 2σ20`, `σ` at `ddof=0` | 20 |
| 12 | `rv20` | `std(r, 20)·√252`, `ddof=1` | 20 |
| 13 | `vol_z` | `(V_t − mean20(V)) / std20(V)` | 20 |

The label is `y_t = ln(C_{t+1} / C_t)`, the *next* day's return, so every feature
row is paired with a return that had not happened when the row was formed.
"""),
        code("""
from src.features import FEATURE_WARMUP, compute_features

frame = compute_features(dbs)
print(f"prices rows           : {len(dbs)}")
print(f"feature rows          : {len(frame)}   ({FEATURE_WARMUP} warm-up days plus NaN rows dropped)")
print(f"feature columns       : {len(FEATURES)} + label_date + {TARGET_COL}")
print(f"date span of features : {frame[DATE_COL].min().date()} → {frame[DATE_COL].max().date()}")

frame[FEATURES].describe().T[["mean", "std", "min", "max"]].round(4)
"""),
        md("""
**Causality check.** The cheapest way to leak the future into a feature is an
accidental centred window or a `shift(-k)`. The test truncates the price series at
an arbitrary date, recomputes every feature, and asserts that the overlapping rows
are bit-identical. If a feature used any future price, truncation would change it.

This is a stronger statement than "we looked at the code": it holds for every
feature simultaneously, at any truncation point.
"""),
        code("""
cutoff = pd.Timestamp("2021-06-30")
truncated = compute_features(dbs[pd.to_datetime(dbs[DATE_COL]) <= cutoff])

last_common = truncated[DATE_COL].max()
overlap = frame[frame[DATE_COL] <= last_common]
assert len(overlap) == len(truncated), "unexpected row alignment"

max_abs_diff = float(
    np.abs(overlap[FEATURES].to_numpy() - truncated[FEATURES].to_numpy()).max()
)
print(f"overlapping rows checked : {len(overlap):,}")
print(f"truncation point         : {last_common.date()}")
print(f"maximum absolute change  : {max_abs_diff:.3e}")
assert max_abs_diff == 0.0, "a feature depends on future prices"
print("PASS — no feature changes when the future is removed")
"""),
        md("""
**Standardisation happens inside each fold, never globally.** The winsorisation
bounds and the z-score statistics are computed on the selection-training rows only
and then applied unchanged to the validation and test rows. Fitting a scaler on
the full sample is one of the most common silent leaks in financial ML; the fold
runner performs this step itself, and the cell below shows the parameters for one
fold so the boundary is visible.
"""),
        code("""
from src.features import fit_scaler, transform_scaler
from src.splits import get_fold, make_folds_from_frame

folds = make_folds_from_frame(frame)
demo_fold = get_fold(folds, "r1f1")

selection_rows = frame.iloc[demo_fold.select_train_idx]
scaler = fit_scaler(selection_rows[FEATURES])
scaled_selection = transform_scaler(selection_rows[FEATURES], scaler)

pd.DataFrame({
    "winsor low": scaler.lower.round(3),
    "winsor high": scaler.upper.round(3),
    "mean": scaler.mean.round(3),
    "std": scaler.std.round(3),
    "scaled mean": pd.Series(scaled_selection.mean(), index=FEATURES).round(6),
}).rename_axis("feature")
"""),
    ]


# ---------------------------------------------------------------------------
# Section 4 — splits
# ---------------------------------------------------------------------------
def section4() -> list:
    return [
        md("""
## 4. Walk-forward splits

The split is expanding and chronological: nothing is shuffled, and a test year is
never seen during training. Each fold has two stages, which is what keeps
hyper-parameter selection honest:

1. **Selection.** Train on everything up to `Y−2`, validate on year `Y−1`, and
   pick the hyper-parameters with the lowest validation RMSE.
2. **Final fit.** Refit on everything up to `Y−1` (validation year now included)
   and predict year `Y`.

**Purge.** A row is assigned to a fold by its *label date* — the date of the return
it predicts — not by the date of its features. The last training row's label is
the first trading day of the test year, so assigning by feature date would leak
exactly one test observation. `src/splits.py` asserts the three boundary
conditions on every fold it builds; the cell below rebuilds them and prints the
resulting windows.
"""),
        code("""
from src.splits import fold_summary

counts = fold_summary(folds, frame["label_date"]).set_index("fold_id")
windows = pd.DataFrame([
    {
        "selection train ends": fold.select_train_end.date(),
        "validation year": f"{fold.val_start.date()} → {fold.val_end.date()}",
        "final train ends": fold.final_train_end.date(),
        "test year": f"{fold.test_start.date()} → {fold.test_end.date()}",
    }
    for fold in folds
], index=[fold.fold_id for fold in folds])
pd.concat([counts[["n_select_train", "n_val", "n_train", "n_test"]], windows], axis=1)
"""),
        md("""
The picture below makes the same structure visible. On the left, the selection
stage: a long training block followed by one validation year. On the right, the
final fit: the validation year is absorbed into training and the test year follows
immediately. Both panels move right by one year per fold, which is why the design
is *expanding* rather than sliding.
"""),
        code("""
figure, (ax_select, ax_final) = plt.subplots(1, 2, figsize=(13, 5.2))
positions = np.arange(len(folds))
origin = pd.Timestamp("2010-01-01")

def draw(axis, start, end, color: str) -> None:
    left = mdates.date2num(pd.Timestamp(start))
    axis.barh(positions, mdates.date2num(pd.Timestamp(end)) - left,
              left=left, height=0.5, color=color)

for fold in folds:
    draw(ax_select, origin, fold.select_train_end, "#9bb7d4")
    draw(ax_select, fold.val_start, fold.val_end, "#d98c5f")
    draw(ax_final, origin, fold.final_train_end, "#9bb7d4")
    draw(ax_final, fold.test_start, fold.test_end, "#5c8a5c")

for axis, title in [(ax_select, "Stage 1 — select hyper-parameters"),
                    (ax_final, "Stage 2 — refit and predict")]:
    axis.set_yticks(positions)
    axis.set_yticklabels([fold.fold_id for fold in folds])
    axis.xaxis.set_major_locator(mdates.YearLocator(2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.3)
    axis.invert_yaxis()

handles = [plt.Rectangle((0, 0), 1, 1, color=colour)
           for colour in ("#9bb7d4", "#d98c5f", "#5c8a5c")]
figure.legend(handles, ["training window", "validation year", "test year"],
              loc="lower center", ncol=3, frameon=False)
figure.suptitle("Expanding walk-forward folds (label dates; 2026 holdout excluded)")
figure.tight_layout(rect=(0, 0.06, 1, 1))
plt.show()
"""),
        md("""
**Why nine folds rather than a single split.** Nine test years span several
regimes — a low-rate bull market, the 2020 crash, and the 2022 rate shock — so a
result that only holds in one of them is visible as such in the per-fold table of
Section 7.2. The cost is that each fold's test set is one year of daily data
(about 250 observations), which is small enough that differences between models
need a formal test rather than a ranking. Section 7.4 supplies that test.
"""),
    ]


# ---------------------------------------------------------------------------
# Section 5 — method
# ---------------------------------------------------------------------------
def section5() -> list:
    return [
        md("""
## 5. Method — DENFIS-inspired evolving Takagi–Sugeno model

The model has three parts, each with one job: a clustering rule decides *where* in
the input space a local model is needed, a fuzzy membership function decides *how
much* each local model contributes to a prediction, and a weighted least-squares
step fits *what* each local model predicts.

### 5.1 Evolving clustering (ECM)

Rules are created online from the stream rather than fixed in advance. Each rule
`i` keeps a centre `c_i` and a radius `R_i` in the standardised feature space.
For a new input `x`, the normalised distance to rule `i` is

$$ d_i(x) = \\frac{\\|x - c_i\\|}{R_i} , $$

and a rule is considered reachable when `d_i(x) ≤ 1`. The closest rule is updated
by moving its centre towards the observation and growing its radius by the same
amount; if no rule is reachable, a new rule is created with `c = x` and
`R = Dthr`. The threshold `Dthr` is therefore the resolution of the partition: small
values give many small local models, large values give few broad ones. This is the
evolving-clustering mechanism of Song and Kasabov (2001), used here as an
antecedent generator.

Two departures from the original algorithm are deliberate and stated as such. A
hard cap of 200 rules bounds runtime and memory on a long stream; and an optional
support-based merge combines two rules when their centres lie within
`merge_factor · Dthr`. Both are disclosed again in Section 8.5 as limitations of
the comparison, not as improvements.

### 5.2 Gaussian membership and first-order TSK inference

Each rule contributes a Gaussian membership

$$ \\mu_i(x) = \\exp\\left(-\\frac{\\|x - c_i\\|^2}{2\\sigma_i^2}\\right), \\qquad \\sigma_i = \\frac{R_i}{2}, $$

and the prediction is the normalised weighted sum of the rules' local linear
models:

$$ \\hat{y}(x) = \\frac{\\sum_i \\mu_i(x)\\, (a_i^\\top x + b_i)}{\\sum_i \\mu_i(x)} . $$

First-order consequents let a rule extrapolate a local trend instead of returning a
constant, which matters because the target is a return whose local level changes
slowly. The normalisation makes the output a convex combination of local
predictions, so an input far from every centre still produces a bounded forecast.

### 5.3 Weighted recursive least squares

The consequent parameters of rule `i` are estimated by RLS on the samples it
dominates, with forgetting factor `λ`:

$$ P_i \\leftarrow \\frac{1}{\\lambda}\\left(P_i - \\frac{P_i \\phi \\phi^\\top P_i}{\\lambda + \\phi^\\top P_i \\phi}\\right), \\qquad a_i \\leftarrow a_i + P_i \\phi \\,(y - \\phi^\\top a_i), $$

where `φ = [x; 1]`. Setting `λ = 1` gives ordinary recursive least squares;
smaller values discount older evidence. The covariance matrix is reset when it
becomes numerically unsafe, and the prediction is clipped to a fixed band so a
single ill-conditioned update cannot produce an absurd return forecast.

### 5.4 Frozen versus online

The two reported variants differ in one place, and only one:

* **`denfis_frozen`** — antecedents and consequents are fitted once on the final
  training window. The test year is predicted with a model that does not change.
* **`denfis_online`** — after each test day's prediction is recorded, the realised
  label is used for one `partial_fit` update. `evolve_rules=True` allows the
  partition to keep growing during the test year.

The ordering is what makes the online variant legitimate: predict first, learn
afterwards. The protocol requires `predict` to be a pure read of model state, and
the contract test below checks exactly that.
"""),
        md("""
### 5.5 Implementation

The full implementation is embedded below rather than imported, so it can be read
in place. It is the file that produced every DENFIS number in this report
(`src/models/denfis.py`); the notebook copies it verbatim at build time.
"""),
        code((ROOT / "src/models/denfis.py").read_text(encoding="utf-8")),
        md("""
Three contract properties are checked on synthetic data: the model must actually
learn a non-linear surface, repeat identically under the same seed, and leave its
state untouched when `predict` is called. The last one is what makes the online
variant's predict-then-update ordering trustworthy.
"""),
        code("""
rng = np.random.default_rng(GLOBAL_SEED)
X_syn = rng.normal(size=(600, 2))
y_syn = np.sin(1.5 * X_syn[:, 0]) + 0.5 * X_syn[:, 1] + 0.1 * rng.normal(size=600)

model_a = DENFIS(dthr=0.4, lam=1.0, prediction_clip=None, seed=GLOBAL_SEED)
model_a.fit(X_syn[:450], y_syn[:450])
prediction_a = model_a.predict(X_syn[450:])

lr = np.sqrt(np.mean((prediction_a - y_syn[450:]) ** 2))
baseline = np.sqrt(np.mean((y_syn[450:] - y_syn[:450].mean()) ** 2))
print(f"rules learned        : {model_a.n_rules_}")
print(f"holdout RMSE         : {lr:.4f}")
print(f"constant-mean RMSE   : {baseline:.4f}")
print(f"explained ratio      : {1 - lr / baseline:.1%}")
assert lr < 0.5 * baseline, "the model does not learn the synthetic surface"

# Determinism: same seed, same data, same rule set and prediction.
model_b = DENFIS(dthr=0.4, lam=1.0, prediction_clip=None, seed=GLOBAL_SEED)
model_b.fit(X_syn[:450], y_syn[:450])
np.testing.assert_array_equal(prediction_a, model_b.predict(X_syn[450:]))
print("determinism          : PASS (identical predictions)")

# Purity: predict must not change the model.
before = model_a.predict(X_syn[450:])
for _ in range(5):
    model_a.predict(X_syn[450:])
np.testing.assert_array_equal(before, model_a.predict(X_syn[450:]))
print("predict purity       : PASS (state unchanged after repeated calls)")
"""),
        md("""
### 5.6 Cost

For `n` training rows, `d` features and `R` rules, one ECM pass costs
`O(n · R · d)` distance evaluations, and the RLS updates cost
`O(n · d²)` per rule touched. With the protocol's cap of 200 rules the per-fold
cost stays in seconds on this dataset — the measured wall-clock time is stored in
each fold's metrics file and summarised in Section 8.1. The operational risk is
not runtime but rule growth: a small `Dthr` on a wide feature distribution will
saturate the cap, which is why `n_rules` is reported next to every result.
"""),
    ]


# ---------------------------------------------------------------------------
# Section 6 — baselines
# ---------------------------------------------------------------------------
def section6() -> list:
    return [
        md("""
## 6. Baselines

Two baselines bracket the comparison from different directions: a strong
non-linear tabular learner and a sequence model.

**XGBoost** (Chen and Guestrin, 2016) sees the same 13 features per row and is
regularised heavily — depths 2–3, learning rates 0.01–0.03, `min_child_weight`
up to 30, `reg_lambda` up to 5.0, with early stopping on the validation year.
The first grid the team tried used shallower regularisation and stopped after a
single tree on the 2016 validation year, which reduces the model to a constant
predictor; the revised grid is the one the protocol records as v1.1.

**LSTM** (Hochreiter and Schmidhuber, 1997) receives 20-day sequences of the same
13 features, one hidden layer of 16 or 32 units, dropout 0.1. The sequence length
is the only structural difference from the other two models, and it is a real one:
the LSTM can in principle recover interactions across time that the row-wise models
cannot express.

The grids are printed from `config.py` so this section cannot drift from the code.
"""),
        code("""
grid_rows = []
for label, grid, keys in [
    ("DENFIS", DENFIS_GRID, ("dthr", "lam")),
    ("XGBoost", XGB_GRID, ("max_depth", "learning_rate", "min_child_weight", "reg_lambda")),
    ("LSTM", LSTM_CONFIGS, ("hidden_size", "seq_len", "dropout")),
]:
    grid_rows.append({
        "model": label,
        "configurations": len(grid),
        "search space": "; ".join(
            f"{key} ∈ {{{', '.join(str(value) for value in sorted({c[key] for c in grid}))}}}"
            for key in keys
        ),
    })
pd.DataFrame(grid_rows).set_index("model")
"""),
        md("""
All three models are selected by the same criterion on the same validation year:
lowest RMSE, with ties broken by fewer DENFIS rules and a larger `Dthr`. Sharing
the selection rule matters, because otherwise a difference in test accuracy could
simply reflect a more lenient tuning budget for one model.

The baseline implementations are listed in the appendix; they follow the same
`fit` / `predict` interface as DENFIS, so the fold runner treats all three
identically.
"""),
    ]


# ---------------------------------------------------------------------------
# Section 7 — results
# ---------------------------------------------------------------------------
def section7() -> list:
    return [
        md("""
## 7. Results

### 7.1 What the numbers mean

* **RMSE / MAE** — forecast error on next-day log returns. A negative `R²` means
  the model is worse than predicting the test-period mean; on daily returns this
  is common and is reported as it is.
* **DA / DA(5bp)** — directional accuracy. The 5 bp variant *excludes* days whose
  realised move is smaller than 5 bp, since a coin flip on a 2 bp move carries no
  information.
* **Sharpe, annual return, maximum drawdown** — a long/flat strategy that holds
  the stock when the forecast exceeds a threshold `τ`, otherwise holds cash. The
  threshold is chosen on the validation year, never on the test year. Each
  position change costs 10 bp one way.
* **Exposure and turnover** — how often the strategy is invested, and how much it
  trades. They are needed to read the Sharpe honestly: a strategy that is invested
  30% of the time earns a different Sharpe for the same signal quality.

Formatting convention in this notebook: RMSE and MAE to 6 decimals (they are
returns of order 10⁻²), `R²` and IC to 4, DA to 4, Sharpe to 3, drawdown and
exposure as percentages to one decimal.
"""),
        code("""
summary = pd.read_csv(RESULTS_DIR / "summary.csv")
pooled = summary[summary["scope"] == "pooled"].set_index("model")

MAIN_MODELS = ["denfis_frozen", "xgboost", "lstm"]

def format_row(model: str) -> dict:
    row = pooled.loc[model]
    return {
        "RMSE": f"{row.rmse:.6f}", "MAE": f"{row.mae:.6f}", "R²": f"{row.r2:.4f}",
        "IC": f"{row.ic:.4f}", "DA": f"{row.da:.4f}", "DA(5bp)": f"{row.da5:.4f}",
        "Sharpe": f"{row.sharpe:.3f}", "ann. return": f"{row.ann_ret:.2%}",
        "max DD": f"{row.max_dd:.1%}", "exposure": f"{row.exposure:.1%}",
        "turnover": f"{row.turnover:.3f}", "n": int(row.n),
    }

main_table = pd.DataFrame(
    [format_row(model) for model in MAIN_MODELS], index=MAIN_MODELS
)
main_table.index.name = "model"
main_table
"""),
        md("""
Buy-and-hold is the benchmark that any long/flat rule has to justify itself
against: a strategy that is invested part of the time and earns a lower Sharpe
than simply holding the stock has not added value. It is computed on the same test
dates from the saved benchmark returns.
"""),
        code("""
from src.evaluation.backtest import annualized_return, max_drawdown, sharpe

reference_preds = pd.concat(
    [pd.read_csv(RESULTS_DIR / "preds/denfis_frozen" / f"r{rep}f{fold}.csv")
     for rep in (1, 2, 3) for fold in (1, 2, 3)],
    ignore_index=True,
).sort_values("date")

bench = reference_preds["bench_ret"].to_numpy(dtype=float)
buy_hold = {
    "Sharpe": f"{sharpe(bench):.3f}",
    "ann. return": f"{annualized_return(bench):.2%}",
    "max DD": f"{max_drawdown(bench):.1%}",
    "exposure": "100.0%",
}
pd.DataFrame(
    [format_row(model) for model in MAIN_MODELS] + [buy_hold],
    index=MAIN_MODELS + ["buy & hold"],
).index.name = "model"
"""),
        md("""
### 7.1b Forecast accuracy against no prediction

The instructor's definition of "better than the baselines" includes a
comparison with **no prediction**: the zero-return forecast, i.e. tomorrow's
adjusted close equals today's. Its RMSE is `sqrt(mean(y**2))` and its MAE is
`mean(|y|)`. Positive skill means the model beats that random-walk forecast;
negative means it is worse.
"""),
        code("""
comparison_models = ["no_prediction"] + MAIN_MODELS
comparison = pooled.reindex(comparison_models)
pd.DataFrame({
    "RMSE": comparison["rmse"].map(lambda x: f"{x:.6f}"),
    "no-pred RMSE": comparison["rmse_zero"].map(lambda x: f"{x:.6f}"),
    "RMSE skill": comparison["rmse_skill"].map(lambda x: f"{x:+.2%}"),
    "MAE": comparison["mae"].map(lambda x: f"{x:.6f}"),
    "no-pred MAE": comparison["mae_zero"].map(lambda x: f"{x:.6f}"),
    "MAE skill": comparison["mae_skill"].map(lambda x: f"{x:+.2%}"),
})
"""),
        md("""
No prediction is a real candidate, not a formality. Every model has to clear
it before its trading rule can be credited with an edge; the table above is the
first test in that chain.

### 7.2 Fold-by-fold

Pooling nine test years hides the fact that a model can win in one regime and lose
badly in another. The per-fold tables below are the honest view: they are what the
statistical test in Section 7.4 operates on.
"""),
        code("""
fold_order = [f"r{rep}f{fold}" for rep in (1, 2, 3) for fold in (1, 2, 3)]
per_fold = summary[summary["scope"] == "fold"].copy()
per_fold["fold_id"] = (
    "r" + per_fold["rep"].astype(int).astype(str)
    + "f" + per_fold["fold"].astype(int).astype(str)
)

def pivot(metric: str, digits: int) -> pd.DataFrame:
    table = per_fold.pivot_table(index="model", columns="fold_id", values=metric)
    return table.loc[MAIN_MODELS + ["denfis_online"], fold_order].round(digits)

print("RMSE by fold")
display(pivot("rmse", 6))
print("Sharpe by fold")
display(pivot("sharpe", 3))
print("Directional accuracy by fold")
display(pivot("da", 4))
"""),
        md("""
### 7.3 Strategy and signal over time

The equity curves pool the nine test years end to end. The dashed line is
buy-and-hold over the same dates.
"""),
        code("""
show_png("results/figures/baseline_equity.png", 11.0, 0.34)
show_png("results/figures/baseline_da.png", 11.0, 0.30)
show_png("results/figures/denfis_online_equity.png", 11.0, 0.34)
"""),
        md("""
### 7.4 Statistical tests

Two questions are separate and need separate tools.

*Is one model more accurate than another?* The Diebold–Mariano test compares the
loss differential series with a Newey–West correction for the autocorrelation
that overlapping forecasts induce. A positive mean loss difference means the first
model is worse.

*Is the trading result distinguishable from luck?* The Sharpe confidence interval
resamples contiguous blocks of days (stationary bootstrap), which preserves some
of the serial dependence in daily returns. An interval that spans zero is the
normal state of affairs for a one-year test set, and it is what this data shows.
"""),
        code("""
import json

with (RESULTS_DIR / "statistical_tests.json").open(encoding="utf-8") as handle:
    stats = json.load(handle)

print(f"Test period: {stats['test_years']}")
print(f"Generated  : {stats['generated_at']}\\n")

ci_rows = []
for model, interval in stats["sharpe_ci"].items():
    low, high = interval["lo"], interval["hi"]
    ci_rows.append({"model": model, "Sharpe": f"{interval['point']:.3f}",
                    "95% CI": f"[{low:.3f}, {high:.3f}]",
                    "spans zero": low <= 0.0 <= high})
pd.DataFrame(ci_rows).set_index("model")
"""),
        code("""
dm_table = pd.DataFrame([
    {
        "A": entry["model_a"],
        "B": entry["model_b"],
        "mean loss diff (A−B)": f"{entry['mean_loss_diff']:+.2e}",
        "DM statistic": f"{entry['dm_stat']:.4f}",
        "p-value": f"{entry['p_value']:.4f}",
        "lag": entry["lag"],
        "outcome": entry["outcome"],
    }
    for entry in stats["dm"]
])
dm_table.set_index(["A", "B"])
"""),
        md("""
Read together, these two tables settle the report's main claim. DENFIS-frozen has
the highest *point estimate* of pooled Sharpe, but its interval spans zero and
runs from slightly negative to well above 1 — the data simply cannot tell its edge
apart from noise. The DM tests point the same way: no pairwise accuracy difference
is significant at any conventional level. We therefore do not claim that DENFIS
forecasts better than XGBoost or LSTM, and we do not claim that the strategy is
profitable. What can be said is narrower and still useful: **the model is
competitive on this task, on the folds where the rule count stays small it is more
interpretable than either baseline, and it takes less drawdown than
buy-and-hold.**

### 7.5 Reproducing one fold from scratch

Everything above reads saved artifacts. To show those artifacts are genuinely the
output of the code in this repository, this section retrains all four models on
fold `r1f1` — selection grid, threshold selection, final fit, test prediction —
and asserts the result matches the committed CSV to within `1e-9`.

Nothing is written to disk: `run_fold` returns the prediction frame, it does not
save it.
"""),
        code("""
import time

from src.evaluation.cv import run_fold
from src.models.baselines import LSTMModel, XGBModel

repro_fold = get_fold(folds, "r1f1")
specs = [
    ("denfis_frozen", lambda params, seed: DENFIS(evolve_rules=False, **params),
     DENFIS_GRID, False, None),
    ("denfis_online", lambda params, seed: DENFIS(evolve_rules=True, **params),
     DENFIS_GRID, True, None),
    ("xgboost", lambda params, seed: XGBModel(params, seed), XGB_GRID, False, None),
    ("lstm", lambda params, seed: LSTMModel(params, seed), LSTM_CONFIGS, False, 20),
]

repro_rows = []
for name, factory, grid, online, sequence_len in specs:
    started = time.perf_counter()
    prediction, metrics = run_fold(
        name, factory, repro_fold, frame, grid,
        seed=GLOBAL_SEED, online=online, sequence_len=sequence_len,
    )
    saved = pd.read_csv(RESULTS_DIR / "preds" / name / "r1f1.csv")

    np.testing.assert_array_equal(
        prediction["date"].to_numpy(), saved["date"].to_numpy()
    )
    max_diff = float(np.max(np.abs(prediction["y_pred"].to_numpy()
                                  - saved["y_pred"].to_numpy())))
    assert max_diff < 1e-9, f"{name}: live run differs from the saved prediction"

    repro_rows.append({
        "model": name,
        "live RMSE": f"{metrics['rmse']:.6f}",
        "saved RMSE": f"{saved.assign(se=lambda d: (d.y_true - d.y_pred) ** 2)['se'].mean() ** 0.5:.6f}",
        "max |Δ prediction|": f"{max_diff:.1e}",
        "seconds": f"{time.perf_counter() - started:.1f}",
    })

pd.DataFrame(repro_rows).set_index("model")
"""),
        md("""
The maximum deviation is at the level of floating-point storage precision
(`~1e-13`), so the committed predictions are reproducible from the committed code
on a different machine. That is the guarantee the protocol asks for, and it is
also the reason the remaining sections can analyse saved artifacts instead of
retraining nine folds per model.

### 7.6 The 2026 holdout

The holdout is a blind period by design: it is read once, after the scheduled
freeze, and never used to select a model, a threshold or a feature. The cell below
prints whatever holdout artifacts exist at build time and says so explicitly when
they do not, rather than filling the gap with an estimate.
"""),
        code("""
holdout_rows = []
for model in MAIN_MODELS + ["denfis_online"]:
    path = RESULTS_DIR / "preds" / model / "holdout.csv"
    if not path.exists():
        holdout_rows.append({"model": model, "holdout": "reserved until the scheduled freeze"})
        continue
    holdout_summary = summary[(summary["model"] == model)
                              & (summary["scope"] == "holdout")]
    if holdout_summary.empty:
        holdout_rows.append({"model": model, "holdout": "prediction file present, metrics missing"})
    else:
        row = holdout_summary.iloc[0]
        holdout_rows.append({
            "model": model,
            "holdout": f"n={int(row.n)}, RMSE {row.rmse:.6f}, "
                       f"DA {row.da:.4f}, Sharpe {row.sharpe:.3f}",
        })
pd.DataFrame(holdout_rows).set_index("model")
"""),
        md("""
Baseline holdout predictions were produced before the DENFIS line was delivered;
they exist but they were not used to choose anything. The DENFIS holdout is run
exactly once, after the nine-fold results are frozen, and its outcome is reported
whatever it shows. Reporting it here before that point would break the blind
protocol.
"""),
        md("""
### 7.7 Implementation check: does the pipeline learn a signal when one exists?

A negative result against no prediction can mean the target is unpredictable, or
it can mean the code is broken. To separate the two, the target is replaced with
a synthetic one built from a real feature:

```text
y_syn = 0.25 * ret_0 + N(0, 0.012)
```

The oracle R² of that planted signal is about 5%. The same folds, scaler,
factories and CV runner are used. If the models beat no prediction here, the
pipeline is learning when a signal exists; the negative result on the real target
is then a property of the data.

Run: `python scripts/synthetic_signal_check.py --models denfis,xgboost,lstm`
"""),
        code("""
synth_path = RESULTS_DIR / "analysis" / "synthetic_signal_check.csv"
if synth_path.exists():
    synth = pd.read_csv(synth_path)
    rows = []
    for model, group in synth.groupby("model"):
        n = group["n"].sum()
        rmse = np.sqrt((group["n"] * group["rmse"] ** 2).sum() / n)
        zero = np.sqrt((group["n"] * group["rmse_zero"] ** 2).sum() / n)
        rows.append({
            "model": model,
            "n": int(n),
            "RMSE": f"{rmse:.6f}",
            "no-pred RMSE": f"{zero:.6f}",
            "RMSE skill": f"{1 - rmse / zero:+.2%}",
        })
    synth_table = pd.DataFrame(rows).set_index("model")
else:
    synth_table = pd.DataFrame(
        {"note": ["run python scripts/synthetic_signal_check.py"]}
    )
synth_table
"""),
    ]


# ---------------------------------------------------------------------------
# Section 8 — analysis
# ---------------------------------------------------------------------------
def section8() -> list:
    return [
        md("""
## 8. Analysis

### 8.1 Parameter sensitivity and model size

DENFIS searches 20 combinations of the cluster radius `Dthr` and the RLS
forgetting factor `λ` on each fold's validation year. If the chosen
hyper-parameters were essentially arbitrary, the model's behaviour would be too
fragile to interpret; if they are stable across folds, the selected model can be
described.

The table below shows, for each fold, which pair won and how many rules the final
model carries. Rule count is the interpretability budget: a model with a dozen
rules can be read, a model at the 200-rule cap cannot.

The result is more mixed than a clean story would suggest. Four folds select the
largest radius (`Dthr = 1.5`) and settle on 11–22 rules; the rest select 0.5, 0.75
or 1.0. **Three folds push the rule count to the cap of 200** (`r1f2`, `r2f3`,
`r3f3`, all with `Dthr ≤ 0.75`). In those folds the partition never stabilised and
the cap — an engineering safeguard, not a modelling choice — is what determines
model size. This directly limits the interpretability claim, and Section 8.5 says
so.
"""),
        code("""
selected_rows = []
for fold_id in fold_order:
    with (RESULTS_DIR / "metrics" / "denfis_frozen" / f"{fold_id}.json").open(encoding="utf-8") as handle:
        record = json.load(handle)
    selected_rows.append({
        "fold": fold_id,
        "test year": record["test_year"],
        "dthr": record["params"]["dthr"],
        "lambda": record["params"]["lam"],
        "validation RMSE": f"{record['validation_rmse']:.6f}",
        "rules": record["n_rules"],
        "cap hits": record["cap_hits"],
        "merges": record["merges"],
        "fit seconds": f"{record['elapsed_seconds']:.1f}",
    })
pd.DataFrame(selected_rows).set_index("fold")
"""),
        md("""
The validation surface is the object being searched. Plotting it for one fold
shows whether the choice is a sharp optimum or a flat plateau — and therefore how
much of the fold-to-fold variation in results is tuning noise.
"""),
        code("""
grid_frame = pd.read_csv(RESULTS_DIR / "analysis/validation_grid.csv")

demo_grid = grid_frame[grid_frame["fold_id"] == "r1f1"].pivot(
    index="dthr", columns="lam", values="val_rmse"
)
figure, axis = plt.subplots(figsize=(6.0, 3.6))
image = axis.imshow(demo_grid.to_numpy(), cmap="viridis", aspect="auto")
axis.set_xticks(range(len(demo_grid.columns)))
axis.set_xticklabels([f"{value:g}" for value in demo_grid.columns])
axis.set_yticks(range(len(demo_grid.index)))
axis.set_yticklabels([f"{value:g}" for value in demo_grid.index])
axis.set_xlabel("forgetting factor λ")
axis.set_ylabel("cluster radius Dthr")
axis.set_title("Validation RMSE, fold r1f1")
for i in range(demo_grid.shape[0]):
    for j in range(demo_grid.shape[1]):
        axis.text(j, i, f"{demo_grid.iloc[i, j]:.5f}", ha="center", va="center",
                  color="white", fontsize=8)
figure.colorbar(image, ax=axis, label="validation RMSE")
figure.tight_layout()
plt.show()
"""),
        code("""
show_png("results/figures/denfis_sensitivity.png", 10.5, 0.42)
"""),
        md("""
### 8.2 Ablations

Two ablations probe whether the two distinctive mechanisms earn their place. Both
are reported as *components of the implementation*, not as causal claims about
the algorithm in general — a caveat that matters for the second one in particular.

* **`no_rls`** replaces the recursive least-squares consequent update with a
  simpler local estimator. It changes the local model *and* its update rule at the
  same time, so it cannot isolate the effect of the forgetting factor alone.
* **`no_merge`** disables rule merging. Most folds never trigger a merge, so this
  ablation is informative mainly on the folds where they do — including `r2f1`,
  which was added specifically because it contains merges.

Four folds are run for each variant (`r1f1`, `r2f1`, `r2f2`, `r3f3`); the table
below is therefore indicative, not a second nine-fold experiment.
"""),
        code("""
ablations = pd.read_csv(RESULTS_DIR / "analysis/ablations.csv")
ablation_table = ablations.pivot_table(index="fold_id", columns="variant",
                                       values=["rmse", "da", "sharpe"])
ablation_table.round({"rmse": 6, "da": 4, "sharpe": 3})
"""),
        code("""
show_png("results/figures/denfis_ablations.png", 10.5, 0.42)
"""),
        md("""
### 8.3 What the rules actually say

Interpretability is the main reason to use a fuzzy rule base here, so it is worth
showing what a fitted model looks like rather than asserting that it is
interpretable.

* The **membership** figure shows each rule's activation across the standardised
  feature space: broad rules dominate most of the input space, narrow ones fire
  only in unusual conditions.
* The **rule evolution** figure shows how the number of rules grows over the test
  year in the online variant. A flat curve means the partition was adequate from
  the start; a rising curve means the model kept discovering new regimes.
* The **contribution** figure decomposes one prediction into its rule terms. It is
  a local, numerical explanation of one forecast — not evidence that the rule is
  economically meaningful.

The chosen case is the largest absolute realised return in fold `r3f3`. It is
selected *after* the fact, purely for illustration, and the model's own prediction
for that day was far too small: this is a case where the model fails, shown
because a failure is more informative than a success.
"""),
        code("""
show_png("results/figures/denfis_memberships.png", 10.5, 0.42)
show_png("results/figures/denfis_rule_evolution.png", 10.5, 0.42)
show_png("results/figures/denfis_case_contributions.png", 10.5, 0.40)

with (RESULTS_DIR / "analysis/high_volatility_case.json").open(encoding="utf-8") as handle:
    case = json.load(handle)
pd.Series({
    "date": case["label_date"],
    "realised return": f"{case['actual']:.4f}",
    "model prediction": f"{case['prediction']:.4f}",
    "selection rule": case["selection"],
})
"""),
        md("""
### 8.4 Transfer from the STI index

A separate, deliberately small experiment asks whether rules learned on the
broader Singapore market transfer to a single bank stock. Three folds and three
protocols are reported in their own directory; **none of these rows enter the
nine-fold main table.**

* **`scratch`** — train on STI, predict DBS, no DBS labels ever used for fitting
  or selection.
* **`zero_shot`** — strict source-only: hyper-parameters and the trading threshold
  are also chosen without DBS labels, using the STI validation year.
* **`fine_tune`** — start from the STI rules, then adapt consequents on DBS
  training data.

Each market is standardised with its own training-only statistics; rules are
matched in standardised space. This is an assumption worth stating explicitly,
because it means a rule centre does not correspond to the same raw price level in
the two markets.
"""),
        code("""
transfer = pd.read_csv(RESULTS_DIR / "transfer/summary.csv")
transfer[["protocol", "fold_id", "test_year", "n", "rmse", "da", "sharpe",
          "n_rules", "test_adaptation"]].sort_values(
    ["protocol", "fold_id"]).set_index(["protocol", "fold_id"]).round(4)
"""),
        md("""
### 8.5 Strengths, weaknesses and how this compares to published work

**What this implementation does well.** Six of the nine folds settle between 11 and
86 rules, and on those the rule base is small enough to read: each prediction
decomposes into a handful of local terms, and the model needs no GPU. Its drawdown
is materially lower than both baselines and buy-and-hold, which is consistent with
a model that stays out of the market when its forecast is near zero.

**Where it is weak.** In the three folds that saturate the 200-rule cap, "an
interpretable fuzzy model" is no longer an accurate description of what was fitted
— the cap, not the data, chose the model size. Forecast error is not better than
the baselines, and the
directional accuracy is indistinguishable from 50%. The model's advantage, if any,
is in *when not to trade*, not in knowing which way the price will move. The
online variant adds an update rule that the frozen one lacks, yet it did not
improve results on this run — plausibly because one year of test data is too short
for adaptation to pay for its extra variance.

**How far this is from published work.** The comparison with the literature is
one of scope, not of score. Published evolving-fuzzy systems are often evaluated
on benchmark regression streams or on higher-frequency data where the
signal-to-noise ratio is more favourable. Deep neuro-fuzzy models such as DCNFIS
operate on image data with vastly more structure per example. We deliberately do
not claim that our RMSE or Sharpe is comparable to any number reported in those
papers: different data, different target, different evaluation. What the
literature supplies here is the mechanism (ECM, TSK, RLS) and the caution that
evolving structure is not automatically an advantage.

**The honest summary.** On this task, with these features and this sample,
DENFIS is a competitive and interpretable model, not a superior one. Section 9
lists what would be needed to make the "superior" claim testable.
"""),
    ]


# ---------------------------------------------------------------------------
# Section 9 — future work
# ---------------------------------------------------------------------------
def section9() -> list:
    return [md("""
## 9. Future work

Four directions follow directly from the limitations above, ordered by how much
they would change the conclusion.

1. **More independent test data.** The binding constraint is not model capacity
   but the number of independent years. Extending to other SGX banks, or to a
   panel of similar markets with shared structure, would give the statistical
   tests something to work with. A nine-year single-stock sample cannot resolve a
   Sharpe difference of the size observed here.

2. **A target the model can actually fit.** Next-day returns are close to
   unpredictable with public daily bars. Volatility, drawdown regime, or
   multi-day horizon returns are all plausible targets where local fuzzy rules
   have a structural reason to help, and where RMSE comparisons would be less
   dominated by noise.

3. **Constrained structure for interpretability.** The current cap and merge rule
   are engineering safeguards. A principled complexity penalty — for example one
   that prices a rule in validation-error units — would turn "few rules" from a
   hope into an objective the model is selected on.

4. **Online evaluation with a longer stream.** The frozen/online comparison needs
   more than one test year per fold to estimate the benefit of adaptation.
   Streaming evaluation across the whole out-of-sample period, with a fixed
   adaptation budget, would test the claim that evolving structure pays for its
   variance.

Each of these is a change to the *experiment*, not to the implementation, which is
the right place to spend effort: the code already reproduces its results exactly.
""")]


# ---------------------------------------------------------------------------
# Section 10 — references
# ---------------------------------------------------------------------------
def section10() -> list:
    return [
        md("""
## 10. References

Numbers below are the citation numbering used throughout the repository
(`docs/refs.md`). Claims in this notebook are attributed to the sources listed
here; no result of another study is reproduced.
"""),
        md("\n".join(
            "### " + line[2:] if line.startswith("# ") else line
            for line in (ROOT / "docs/refs.md").read_text(encoding="utf-8").splitlines()
        )),
    ]


# ---------------------------------------------------------------------------
# Appendix
# ---------------------------------------------------------------------------
def appendix() -> list:
    modules = [
        "src/data.py",
        "src/features.py",
        "src/splits.py",
        "src/models/baselines.py",
        "src/evaluation/metrics.py",
        "src/evaluation/backtest.py",
        "src/evaluation/stats.py",
        "src/evaluation/cv.py",
        "src/transfer.py",
    ]
    cells = [md("""
## Appendix — complete implementation

The DENFIS model itself is inlined in Section 5.5. The remaining modules that
produce the numbers in this report are listed here in full, so the notebook is
self-contained as a record of the method even without the repository.

Every listing below is the exact file content at the git commit printed in
Section 1.
""")]
    for module in modules:
        cells.append(md(listing(module)))
    return cells


def build() -> nbformat.NotebookNode:
    notebook = new_notebook()
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.9"}
    cells = []
    for section in (section0, section1, section2, section3, section4,
                    section5, section6, section7, section8, section9,
                    section10, appendix):
        cells.extend(section())
    notebook.cells = cells
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-exec", action="store_true",
                        help="write the notebook without executing it")
    args = parser.parse_args()

    notebook = build()
    if not args.no_exec:
        from nbclient import NotebookClient
        client = NotebookClient(
            notebook,
            timeout=1800,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
            allow_errors=False,
        )
        client.execute()
    nbformat.write(notebook, OUT)
    print(f"wrote {OUT.relative_to(ROOT)} with {len(notebook.cells)} cells")


if __name__ == "__main__":
    main()
