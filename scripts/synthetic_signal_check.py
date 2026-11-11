"""Implementation check: can the pipeline learn a signal that is known to exist?

The real DBS target is close to unpredictable at the daily horizon, so a
negative result against the no-prediction forecast does not by itself tell us
whether the code is correct.  This script replaces the target with a synthetic
signal built from one of the real features:

    y_syn = SIGNAL * ret_0 + N(0, NOISE)

and runs the exact same folds, scaler, model factories and CV runner.  If the
models beat the no-prediction forecast here, the pipeline is learning when a
signal exists; the negative result on the real target is then an empirical
property of the data, not an implementation bug.

Examples
--------
python scripts/synthetic_signal_check.py --models denfis,xgboost
python scripts/synthetic_signal_check.py --models denfis,xgboost,lstm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    DENFIS_MAX_RULES,
    DENFIS_MERGE_FACTOR,
    RESULTS_DIR,
    TARGET_COL,
)
from src.data import load_prices  # noqa: E402
from src.evaluation.cv import run_fold  # noqa: E402
from src.features import compute_features  # noqa: E402
from src.splits import make_folds_from_frame  # noqa: E402

ANALYSIS_DIR = RESULTS_DIR / "analysis"
SIGNAL = 0.25
NOISE = 0.012
NOISE_SEED = 7

FIXED_PARAMS = {
    "denfis": {"dthr": 1.5, "lam": 1.0},
    "xgboost": {
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "reg_lambda": 1.0,
        "n_estimators": 200,
    },
    "lstm": {"hidden_size": 16, "seq_len": 20, "dropout": 0.1, "epochs": 40},
}
SEQUENCE_LENGTH = {"denfis": None, "xgboost": None, "lstm": 20}


def _factory(name: str):
    if name == "denfis":
        from src.models.denfis import DENFIS

        def build(params, seed):
            return DENFIS(
                **params,
                max_rules=DENFIS_MAX_RULES,
                merge_factor=DENFIS_MERGE_FACTOR,
                seed=seed,
            )

        return build

    from src.models.baselines import LSTMModel, XGBModel

    model_class = {"xgboost": XGBModel, "lstm": LSTMModel}[name]
    return lambda params, seed: model_class(params, seed)


def _synthetic_target(frame: pd.DataFrame, signal: float, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    values = signal * frame["ret_0"].to_numpy(dtype=float)
    values = values + rng.normal(0.0, noise, size=len(frame))
    oracle_r2 = 1.0 - np.var(values - signal * frame["ret_0"].to_numpy(dtype=float)) / np.var(values)
    return values, float(oracle_r2)


def _pooled(rows: list[dict]) -> dict:
    n = sum(row["n"] for row in rows)
    rmse = float(np.sqrt(sum(row["n"] * row["rmse"] ** 2 for row in rows) / n))
    rmse_zero = float(np.sqrt(sum(row["n"] * row["rmse_zero"] ** 2 for row in rows) / n))
    mae = float(sum(row["n"] * row["mae"] for row in rows) / n)
    mae_zero = float(sum(row["n"] * row["mae_zero"] for row in rows) / n)
    return {
        "n": int(n),
        "rmse": rmse,
        "rmse_zero": rmse_zero,
        "rmse_skill": float(1.0 - rmse / rmse_zero),
        "mae": mae,
        "mae_zero": mae_zero,
        "mae_skill": float(1.0 - mae / mae_zero),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="denfis,xgboost",
        help="Comma-separated subset of denfis,xgboost,lstm.",
    )
    parser.add_argument("--signal", type=float, default=SIGNAL)
    parser.add_argument("--noise", type=float, default=NOISE)
    parser.add_argument("--noise-seed", type=int, default=NOISE_SEED)
    args = parser.parse_args()

    names = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = set(names) - set(FIXED_PARAMS)
    if unknown:
        raise SystemExit(f"unknown models: {sorted(unknown)}")

    frame = compute_features(load_prices("D05_SI"))
    y_syn, oracle_r2 = _synthetic_target(frame, args.signal, args.noise, args.noise_seed)
    synth = frame.copy()
    synth[TARGET_COL] = y_syn
    folds = make_folds_from_frame(synth)

    rows: list[dict] = []
    for name in names:
        for fold in folds:
            if fold.fold_id == "holdout":
                continue
            pred, metrics = run_fold(
                name,
                _factory(name),
                fold,
                synth,
                [dict(FIXED_PARAMS[name])],
                seed=42,
                sequence_len=SEQUENCE_LENGTH[name],
            )
            rows.append(
                {
                    "model": name,
                    "fold": fold.fold_id,
                    "n": len(pred),
                    "rmse": metrics["rmse"],
                    "rmse_zero": metrics["rmse_zero"],
                    "rmse_skill": metrics["rmse_skill"],
                    "mae": metrics["mae"],
                    "mae_zero": metrics["mae_zero"],
                    "mae_skill": metrics["mae_skill"],
                }
            )
            print(
                f"{name:10s} {fold.fold_id:6s} "
                f"rmse={metrics['rmse']:.6f} zero={metrics['rmse_zero']:.6f} "
                f"skill={metrics['rmse_skill']:+.2%}"
            )

    detail = pd.DataFrame(rows)
    analysis_dir = ANALYSIS_DIR
    analysis_dir.mkdir(parents=True, exist_ok=True)
    detail_path = analysis_dir / "synthetic_signal_check.csv"
    summary_path = analysis_dir / "synthetic_signal_check.json"

    # Running a subset updates only those models; earlier results are kept so
    # the denfis/xgboost and lstm runs can be done separately.
    if detail_path.exists():
        existing = pd.read_csv(detail_path)
        existing = existing[~existing["model"].isin(names)]
        detail = pd.concat([existing, detail], ignore_index=True)
    detail.to_csv(detail_path, index=False)

    all_names = sorted(detail["model"].unique())
    pooled = {
        name: _pooled(detail[detail["model"] == name].to_dict("records"))
        for name in all_names
    }
    summary = {
        "signal": float(args.signal),
        "noise": float(args.noise),
        "noise_seed": int(args.noise_seed),
        "oracle_r2": oracle_r2,
        "description": "y_syn = signal * ret_0 + N(0, noise); same folds and CV runner",
        "pooled": pooled,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"oracle R2 of the planted signal: {oracle_r2:.3f}")
    table = pd.DataFrame(pooled).T.reset_index().rename(columns={"index": "model"})
    print(table.to_string(index=False))
    print()
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
