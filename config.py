"""Single source of truth for AI6124 Group 18 (protocol v1.0).

Nothing in this file depends on another project module. Every script and
notebook imports constants from here so that folds, features, seeds, and
cost assumptions cannot drift between the two workstreams.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
PREDS_DIR = RESULTS_DIR / "preds"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
PROTOCOL_VERSION = "1.1"  # v1.1: stronger XGBoost regularisation grid (pending A sign-off)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
SYMBOL_DBS = "D05.SI"
SYMBOL_STI = "^STI"
START_DATE = "2010-01-01"
END_DATE = "2026-10-31"

DATE_COL = "Date"
PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

# The 13 features are frozen for the main experiment. Order matters: model
# inputs are positional arrays built from this list.
FEATURES = [
    "ret_0",
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_4",
    "ma5",
    "ma20",
    "ma60",
    "rsi14",
    "macd",
    "boll_b",
    "rv20",
    "vol_z",
]
TARGET_COL = "y"
FEATURE_WARMUP = 60  # longest lookback is MA60

# ---------------------------------------------------------------------------
# Walk-forward folds
# ---------------------------------------------------------------------------
# For test year Y:
#   selection train : 2010-01-01 .. (Y-2)-12-31
#   validation      : (Y-1)-01-01 .. (Y-1)-12-31
#   final train     : 2010-01-01 .. (Y-1)-12-31
#   test            : Y-01-01 .. Y-12-31
FOLD_SPECS = [
    {"fold_id": "r1f1", "rep": 1, "fold": 1, "test_year": 2017},
    {"fold_id": "r1f2", "rep": 1, "fold": 2, "test_year": 2018},
    {"fold_id": "r1f3", "rep": 1, "fold": 3, "test_year": 2019},
    {"fold_id": "r2f1", "rep": 2, "fold": 1, "test_year": 2020},
    {"fold_id": "r2f2", "rep": 2, "fold": 2, "test_year": 2021},
    {"fold_id": "r2f3", "rep": 2, "fold": 3, "test_year": 2022},
    {"fold_id": "r3f1", "rep": 3, "fold": 1, "test_year": 2023},
    {"fold_id": "r3f2", "rep": 3, "fold": 2, "test_year": 2024},
    {"fold_id": "r3f3", "rep": 3, "fold": 3, "test_year": 2025},
]

HOLDOUT_SPEC = {
    "fold_id": "holdout",
    "rep": 0,
    "fold": 0,
    "test_start": "2026-01-01",
    "test_end": "2026-10-31",
    "val_start": "2025-01-01",
    "val_end": "2025-12-31",
    "select_train_end": "2024-12-31",
    "final_train_end": "2025-12-31",
}

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42
HOLDOUT_SEED = 99
BOOTSTRAP_SEED = 123


def fold_seed(rep: int, fold: int) -> int:
    """Seed for a regular fold: r1f1=42, r1f2=43, ..., r3f3=48."""
    return GLOBAL_SEED + 100 * (rep - 1) + (fold - 1)


# ---------------------------------------------------------------------------
# Evaluation / trading
# ---------------------------------------------------------------------------
COST_BPS = 10.0                     # one-way transaction cost, basis points
TAU_GRID = (0.0, 5e-4, 1e-3, 2e-3)  # signal thresholds, log-return units
ANNUALIZATION = 252
DEADBAND = 5e-4                     # directional accuracy deadband (5 bp)

# ---------------------------------------------------------------------------
# Model search spaces (frozen in the protocol)
# ---------------------------------------------------------------------------
DENFIS_DTHR_GRID = (0.25, 0.5, 0.75, 1.0, 1.5)
DENFIS_LAMBDA_GRID = (0.95, 0.98, 0.99, 1.0)
DENFIS_MAX_RULES = 200
DENFIS_MERGE_FACTOR = 0.5

# Parameter dictionaries passed to the DENFIS constructor.  Keys are the
# interface contract with the method workstream (WORKPLAN_A): ``dthr`` and
# ``lam`` are the two searched knobs; ``max_rules``, ``merge_factor`` and
# ``seed`` are fixed from the constants above.
DENFIS_GRID = [
    {"dthr": dthr, "lam": lam}
    for dthr in DENFIS_DTHR_GRID
    for lam in DENFIS_LAMBDA_GRID
]

# Daily returns carry little signal, so the first grid version stopped at one
# tree on the 2016 validation year.  The revised grid searches stronger
# regularisation as well; validation RMSE then bottoms out after 100+ trees.
XGB_GRID = [
    {
        "max_depth": depth,
        "learning_rate": lr,
        "min_child_weight": min_child_weight,
        "reg_lambda": reg_lambda,
    }
    for depth in (2, 3)
    for lr in (0.01, 0.03)
    for min_child_weight in (5, 30)
    for reg_lambda in (1.0, 5.0)
]

LSTM_CONFIGS = [
    {"hidden_size": hidden, "seq_len": 20, "dropout": 0.1}
    for hidden in (16, 32)
]
