"""End-to-end sanity check for the feature and fold pipeline on cached data.

Run from the project root:
    python scripts/check_pipeline.py

Prints the feature frame shape, fold table, and a naive zero-prediction
baseline.  Writes docs/feature_fold_summary.csv and docs/naive_zero.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FEATURES, TARGET_COL  # noqa: E402
from src.data import load_prices  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.features import compute_features  # noqa: E402
from src.splits import fold_summary, make_folds_from_frame  # noqa: E402


def main() -> None:
    prices = load_prices("D05_SI")
    frame = compute_features(prices)
    folds = make_folds_from_frame(frame)
    summary = fold_summary(folds, frame["label_date"])

    print(f"prices rows : {len(prices):,}")
    print(f"features    : {len(frame):,}  ({frame['Date'].min().date()} .. {frame['Date'].max().date()})")
    print(f"nan features: {int(frame[FEATURES].isna().sum().sum())}")
    print(f"folds       : {len(folds)}")
    print()
    print(summary.to_string(index=False))

    rows = []
    for fold in folds:
        test = frame.iloc[fold.test_idx]
        y_true = test[TARGET_COL].to_numpy()
        y_zero = np.zeros_like(y_true)
        metrics = compute_metrics(y_true, y_zero)
        metrics.update({"fold_id": fold.fold_id, "test_year": fold.test_year})
        rows.append(metrics)
    naive = pd.DataFrame(rows).set_index("fold_id")
    print()
    print("Naive zero-prediction baseline")
    print(naive[["n", "rmse", "rmse_zero", "mae", "mae_zero"]].to_string())
    print(
        "Directional accuracy is undefined for a zero forecast: sign(0) = 0, "
        "so it only counts exactly-zero returns."
    )

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    summary.to_csv(docs / "feature_fold_summary.csv", index=False)
    naive.to_csv(docs / "naive_zero.csv")
    print(f"\nWrote {docs / 'feature_fold_summary.csv'}")
    print(f"Wrote {docs / 'naive_zero.csv'}")


if __name__ == "__main__":
    main()
