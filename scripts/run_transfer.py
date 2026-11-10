"""Three held-out-fold STI -> DBS protocols using the shared CV evaluator.

Writes results/transfer/{preds,metrics}/{protocol}/{fold}.{csv,json}.
Never mixes partial-fold transfer evidence with nine-fold main results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DENFIS_GRID, FEATURES, TARGET_COL, fold_seed
from src.evaluation.cv import run_fold
from src.features import fit_scaler, transform_scaler
from src.models.denfis import DENFIS
from src.splits import make_folds_from_frame
from src.transfer import load_transfer_frames


class TransferDENFIS(DENFIS):
    """Use precomputed training-only source windows; freeze during test."""

    def __init__(self, source_windows, protocol, **kwargs):
        super().__init__(**kwargs)
        self.source_windows = source_windows
        self.protocol = protocol

    def fit(self, X, y):
        if self.protocol == "scratch":
            return super().fit(X, y)
        source_X, source_y = self.source_windows[len(X)]
        super().fit(source_X, source_y)
        if self.protocol == "fine_tune":
            self.evolve_rules = True
            for x, target in zip(X, y):
                self.partial_fit(x, float(target))
            self.evolve_rules = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="r1f1,r2f2,r3f3")
    parser.add_argument("--protocols", default="scratch,zero_shot,fine_tune")
    args = parser.parse_args()
    frames = load_transfer_frames()
    dbs, sti = frames["DBS"], frames["STI"]
    folds = {f.fold_id: f for f in make_folds_from_frame(dbs)}
    requested = args.folds.split(",")
    if any(fid == "holdout" or fid not in folds for fid in requested):
        raise ValueError("transfer supports regular folds only")
    output = ROOT / "results" / "transfer"
    protocols = args.protocols.split(",")
    if set(protocols) - {"scratch", "zero_shot", "fine_tune"}:
        raise ValueError("unknown transfer protocol")
    rows = []
    for fid in requested:
        fold = folds[fid]
        source_windows = {}
        for idx, cutoff in ((fold.select_train_idx, fold.select_train_end),
                            (fold.train_idx, fold.final_train_end)):
            source = sti.iloc[idx]
            assert source["label_date"].max() <= cutoff
            source_windows[len(idx)] = (
                transform_scaler(source[FEATURES], fit_scaler(source[FEATURES])),
                source[TARGET_COL].to_numpy(dtype=np.float64),
            )
        for protocol in protocols:
            def factory(params, seed):
                return TransferDENFIS(source_windows, protocol, **params, seed=seed)
            preds, metrics = run_fold(
                f"denfis_transfer_{protocol}", factory, fold, dbs, DENFIS_GRID,
                seed=fold_seed(fold.rep, fold.fold),
                selection_frame=sti if protocol == "zero_shot" else None,
            )
            metrics.update({"protocol": protocol,
                            "source_select_end": str(sti.iloc[fold.select_train_idx]["label_date"].max()),
                            "source_final_end": str(sti.iloc[fold.train_idx]["label_date"].max()),
                            "test_start": str(dbs.iloc[fold.test_idx]["label_date"].min()),
                            "market_scaling": "separate training-only winsorisation/z-score",
                            "test_adaptation": False})
            metrics["selection_market"] = "STI" if protocol == "zero_shot" else "DBS"
            for kind in ("preds", "metrics"):
                (output / kind / protocol).mkdir(parents=True, exist_ok=True)
            preds.to_csv(output / "preds" / protocol / f"{fid}.csv", index=False)
            (output / "metrics" / protocol / f"{fid}.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8")
            rows = []
            for path in sorted((output / "metrics").glob("*/*.json")):
                record = json.loads(path.read_text())
                rows.append({k: v for k, v in record.items() if isinstance(v, (str, int, float, bool))})
            pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
            print(protocol, fid, metrics["rmse"], metrics["sharpe"], flush=True)


if __name__ == "__main__":
    main()
