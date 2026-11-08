"""Run a benchmark model over selected folds.

Examples
--------
python scripts/run_baseline.py --model xgboost --folds r1f1
python scripts/run_baseline.py --model xgboost --folds all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import LSTM_CONFIGS, XGB_GRID  # noqa: E402
from src.data import load_prices  # noqa: E402
from src.evaluation.cv import run_model_on_folds  # noqa: E402
from src.features import compute_features  # noqa: E402
from src.models.baselines import LSTMModel, XGBModel  # noqa: E402
from src.splits import make_folds_from_frame  # noqa: E402

MODELS = {
    "xgboost": (lambda params, seed: XGBModel(params, seed), XGB_GRID, False, None),
    "lstm": (lambda params, seed: LSTMModel(params, seed), LSTM_CONFIGS, False, 20),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), default="xgboost")
    parser.add_argument(
        "--folds",
        default="r1f1",
        help="Comma-separated fold ids, or 'all' for the nine regular folds.",
    )
    args = parser.parse_args()

    prices = load_prices("D05_SI")
    frame = compute_features(prices)
    folds = make_folds_from_frame(frame)

    fold_ids = None if args.folds == "all" else [x.strip() for x in args.folds.split(",")]
    factory, grid, online, sequence_len = MODELS[args.model]

    summary = run_model_on_folds(
        args.model,
        factory,
        folds,
        frame,
        grid,
        online=online,
        fold_ids=fold_ids,
        sequence_len=sequence_len,
    )
    print()
    print(summary[summary["model"] == args.model].to_string(index=False))


if __name__ == "__main__":
    main()
