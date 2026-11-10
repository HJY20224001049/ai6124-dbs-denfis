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

from config import (  # noqa: E402
    DENFIS_GRID,
    DENFIS_MAX_RULES,
    DENFIS_MERGE_FACTOR,
    LSTM_CONFIGS,
    XGB_GRID,
)
from src.data import load_prices  # noqa: E402
from src.evaluation.cv import run_model_on_folds  # noqa: E402
from src.features import compute_features  # noqa: E402
from src.models.baselines import LSTMModel, XGBModel  # noqa: E402
from src.splits import make_folds_from_frame  # noqa: E402


def _denfis_factory(evolve_rules: bool = False):
    """Build the DENFIS factory lazily so baselines run before A lands.

    Interface contract (WORKPLAN_A): ``DENFIS(dthr, lam, max_rules,
    merge_factor, seed, evolve_rules)``.  The CV runner handles the
    predict-then-update order for the online variant; ``evolve_rules`` only
    controls whether ECM keeps growing rules during ``partial_fit``.
    """
    def factory(params, seed):
        try:
            from src.models.denfis import DENFIS
        except ImportError as exc:  # noqa: BLE001
            raise SystemExit(
                "src/models/denfis.py is not available yet. "
                "The method workstream must deliver DENFIS(dthr, lam, max_rules, "
                "merge_factor, seed, evolve_rules) before this model can run."
            ) from exc
        return DENFIS(
            **params,
            max_rules=DENFIS_MAX_RULES,
            merge_factor=DENFIS_MERGE_FACTOR,
            seed=seed,
            evolve_rules=evolve_rules,
        )

    return factory


MODELS = {
    "xgboost": (lambda params, seed: XGBModel(params, seed), XGB_GRID, False, None),
    "lstm": (lambda params, seed: LSTMModel(params, seed), LSTM_CONFIGS, False, 20),
    "denfis_frozen": (_denfis_factory(), DENFIS_GRID, False, None),
    "denfis_online": (_denfis_factory(evolve_rules=True), DENFIS_GRID, True, None),
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
