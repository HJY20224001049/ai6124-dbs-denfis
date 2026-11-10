"""Reproduce method diagnostics; only validation records inform sensitivity.

Run after both nine-fold model runs. Ablations inherit the frozen-selected
parameters, and use shared run_fold rather than a second evaluation loop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import FEATURES, TARGET_COL, fold_seed
from src.data import load_prices
from src.evaluation.cv import run_fold
from src.features import compute_features, fit_scaler, transform_scaler
from src.models.denfis import DENFIS
from src.splits import make_folds_from_frame


def main():
    output = ROOT / "results" / "analysis"
    figures = ROOT / "results" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    frame = compute_features(load_prices("D05_SI"))
    folds = {f.fold_id: f for f in make_folds_from_frame(frame)}
    representative = ["r1f1", "r2f2", "r3f3"]
    sensitivity, ablations = [], []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    # Include the merge-active fold, otherwise disabling merge is an identity
    # operation in all three planned representative fits.
    for fid in ("r1f1", "r2f1", "r2f2", "r3f3"):
        metadata = json.loads((ROOT / "results/metrics/denfis_frozen" / f"{fid}.json").read_text())
        for row in metadata["search_results"]:
            sensitivity.append({"fold_id": fid, **row["params"],
                                "val_rmse": row["val_rmse"], "n_rules": row["n_rules"]})
        fold = folds[fid]
        params = metadata["params"]
        for variant, extra in (("no_rls", {"rls": False}),
                               ("no_merge", {"merge_factor": 0})):
            def factory(p, seed):
                return DENFIS(**p, **extra, seed=seed)
            preds, metrics = run_fold(variant, factory, fold, frame, [params],
                                      seed=fold_seed(fold.rep, fold.fold))
            (output / "preds" / variant).mkdir(parents=True, exist_ok=True)
            (output / "metrics" / variant).mkdir(parents=True, exist_ok=True)
            preds.to_csv(output / "preds" / variant / f"{fid}.csv", index=False)
            (output / "metrics" / variant / f"{fid}.json").write_text(json.dumps(metrics, indent=2))
            ablations.append({"fold_id": fid, "variant": variant,
                              **{k: metrics[k] for k in ("rmse", "da", "sharpe", "n_rules", "merges")}})
        for variant in ("denfis_frozen", "denfis_online"):
            row = json.loads((ROOT / "results/metrics" / variant / f"{fid}.json").read_text())
            ablations.append({"fold_id": fid, "variant": variant,
                              **{k: row[k] for k in ("rmse", "da", "sharpe", "n_rules", "merges")}})
    data = pd.DataFrame(sensitivity)
    data.to_csv(output / "validation_grid.csv", index=False)
    pd.DataFrame(ablations).to_csv(output / "ablations.csv", index=False)
    for ax, parameter in zip(axes, ("dthr", "lam")):
        for fid in representative:
            profile = data[data.fold_id == fid].groupby(parameter).val_rmse.min()
            ax.plot(profile.index, profile.values, marker="o", label=fid)
        ax.set(xlabel=parameter, ylabel="Validation RMSE",
               title="Profile minimum over the other parameter")
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "denfis_sensitivity.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for fid in folds:
        path = ROOT / "results/metrics/denfis_online" / f"{fid}.json"
        if path.exists():
            history = json.loads(path.read_text())["rule_history"]
            ax.plot(history, label=fid)
    ax.set(xlabel="Realised test labels processed", ylabel="Rule count",
           title="Online rule evolution (initial count after final training)")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(figures / "denfis_rule_evolution.png", dpi=150)
    plt.close(fig)

    fid = "r3f3"
    fold = folds[fid]
    metadata = json.loads((ROOT / "results/metrics/denfis_frozen" / f"{fid}.json").read_text())
    train, test = frame.iloc[fold.train_idx], frame.iloc[fold.test_idx]
    scaler = fit_scaler(train[FEATURES])
    model = DENFIS(**metadata["params"], seed=fold_seed(fold.rep, fold.fold))
    model.fit(transform_scaler(train[FEATURES], scaler), train[TARGET_COL].to_numpy())
    X = transform_scaler(test[FEATURES], scaler)
    h = model.firing_strengths(X)
    top = np.argsort(h.mean(axis=0))[-3:]
    np.savez_compressed(output / "r3f3_rules.npz", centers=model.centers_,
                        radii=model.radii_, coefficients=model.coef_,
                        support=model.support_, average_firing=h.mean(axis=0))
    grid = np.linspace(-3, 3, 200)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for index in top:
        sigma = np.sqrt(len(FEATURES)) * max(model.radii_[index], 0.1*model.dthr, 1e-6)
        membership = np.exp(-0.5*((grid-model.centers_[index, 0])/sigma)**2)
        ax.plot(grid, membership, label=f"rule {index} (mean h={h[:, index].mean():.3f})")
    ax.set(xlabel="ret_0 (winsorised z-score)", ylabel="Coordinate membership",
           title="Three most active r3f3 rules: marginal Gaussian sets")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "denfis_memberships.png", dpi=150)
    plt.close(fig)

    case = int(np.argmax(np.abs(test[TARGET_COL].to_numpy())))
    local = model.coef_ @ np.r_[1., X[case]]
    contribution = h[case] * local
    cases = pd.DataFrame({"rule": np.arange(model.n_rules_), "firing": h[case],
                          "local_output": local, "contribution": contribution})
    cases.to_csv(output / "high_volatility_contributions.csv", index=False)
    (output / "high_volatility_case.json").write_text(json.dumps({
        "label_date": str(test.iloc[case].label_date),
        "selection": "largest absolute realised return in r3f3; post-hoc explanation only",
        "actual": float(test.iloc[case][TARGET_COL]),
        "prediction": float(model.predict(X[case:case+1])[0]),
        "unclipped_sum": float(contribution.sum()),
    }, indent=2))
    strongest = cases.reindex(cases.contribution.abs().sort_values(ascending=False).index).head(8)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(strongest.rule.astype(str), strongest.contribution)
    ax.set(xlabel="Rule index", ylabel="Contribution (log return)",
           title=f"Post-hoc high-volatility case: {test.iloc[case].label_date.date()}")
    fig.tight_layout()
    fig.savefig(figures / "denfis_case_contributions.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for variant, part in pd.DataFrame(ablations).groupby("variant"):
        ax.plot(part.fold_id, part.rmse, marker="o", label=variant)
    ax.set(xlabel="Representative test fold", ylabel="Test RMSE",
           title="Ablations inherit frozen-selected parameters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "denfis_ablations.png", dpi=150)
    plt.close(fig)

    rng = np.random.default_rng(42)
    demo = rng.normal(size=(250, 2))
    m = DENFIS(dthr=.4, prediction_clip=None)
    m.fit(demo, np.sin(demo[:, 0]))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(*demo.T, s=8, alpha=.4)
    ax.scatter(*m.centers_.T, marker="x", color="red")
    for center, radius in zip(m.centers_, m.radii_):
        ax.add_patch(plt.Circle(center, radius*np.sqrt(2), fill=False, alpha=.5))
    ax.set(xlabel="Synthetic x0", ylabel="Synthetic x1", title="ECM centres and radii")
    ax.set_aspect("equal")
    (ROOT / "docs/figs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "docs/figs/ecm_demo.png", dpi=150)
    plt.close(fig)
    print("Wrote", output, flush=True)


if __name__ == "__main__":
    main()
