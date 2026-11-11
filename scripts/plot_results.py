"""Baseline comparison plots from persisted predictions.

Run after the baseline folds:
    python scripts/plot_results.py

Writes results/figures/baseline_equity.png and baseline_da.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FIGURES_DIR, METRICS_DIR, PREDS_DIR  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "legend.fontsize": 8,
        "figure.autolayout": True,
    }
)

COLORS = {
    "no_prediction": "#6B7280",
    "xgboost": "#94510F",
    "lstm": "#0E6B5A",
    "denfis_frozen": "#1F4E79",
    "denfis_online": "#7A3E8C",
}


def _models() -> list[str]:
    return sorted(p.name for p in PREDS_DIR.iterdir() if p.is_dir() and any(p.glob("*.csv")))


def _regular_predictions(model: str) -> pd.DataFrame:
    frames = []
    for path in sorted((PREDS_DIR / model).glob("*.csv")):
        if path.stem == "holdout":
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def plot_equity(models: list[str], suffix: str = "baseline", online: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bench = None
    for model in models:
        preds = _regular_predictions(model)
        if preds.empty:
            continue
        dates = pd.to_datetime(preds["date"])
        equity = (1.0 + preds["strat_ret"]).cumprod()
        ax.plot(dates, equity, label=model, color=COLORS.get(model, None), linewidth=1.0)
        if bench is None:
            bench = (1.0 + preds["bench_ret"]).cumprod()
            ax.plot(dates, bench, label="buy & hold", color="#16211E", linewidth=0.9, linestyle="--")
    ax.set_title(("Online extension: predict then update" if online else "Frozen comparison, 2017–2025") + " (10 bp one-way cost)")
    ax.set_ylabel("Growth of 1 SGD")
    ax.set_xlabel("Date")
    ax.legend(frameon=False, ncol=2)
    path = FIGURES_DIR / f"{suffix}_equity.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_directional_accuracy(models: list[str], suffix: str = "baseline") -> Path:
    folds = [f"r{rep}f{fold}" for rep in (1, 2, 3) for fold in (1, 2, 3)]
    values = {}
    for model in models:
        row = []
        for fold in folds:
            path = METRICS_DIR / model / f"{fold}.json"
            if not path.exists():
                row.append(np.nan)
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            row.append(data["da"])
        values[model] = row

    x = np.arange(len(folds))
    width = 0.8 / max(len(values), 1)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for offset, (model, row) in enumerate(values.items()):
        ax.bar(
            x - 0.4 + width / 2 + offset * width,
            row,
            width,
            label=model,
            color=COLORS.get(model, None),
        )
    ax.axhline(0.5, color="#16211E", linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Directional accuracy")
    ax.set_title("Directional accuracy by test fold")
    ax.legend(frameon=False, ncol=2)
    path = FIGURES_DIR / f"{suffix}_da.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rmse_skill(models: list[str], suffix: str = "baseline") -> Path:
    """Per-fold RMSE skill against the no-prediction forecast."""
    from src.evaluation.metrics import compute_metrics

    folds = [f"r{rep}f{fold}" for rep in (1, 2, 3) for fold in (1, 2, 3)]
    values = {}
    for model in models:
        row = []
        for fold in folds:
            path = PREDS_DIR / model / f"{fold}.csv"
            if not path.exists():
                row.append(np.nan)
                continue
            data = pd.read_csv(path)
            row.append(compute_metrics(data["y_true"], data["y_pred"])["rmse_skill"])
        values[model] = row

    x = np.arange(len(folds))
    width = 0.8 / max(len(values), 1)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for offset, (model, row) in enumerate(values.items()):
        ax.bar(
            x - 0.4 + width / 2 + offset * width,
            row,
            width,
            label=model,
            color=COLORS.get(model, None),
        )
    ax.axhline(0.0, color="#16211E", linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylabel("RMSE skill vs no prediction")
    ax.set_title("Does the model beat the no-prediction forecast?")
    ax.legend(frameon=False, ncol=2)
    path = FIGURES_DIR / f"{suffix}_rmse_skill.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    models = _models()
    if not models:
        raise SystemExit("No predictions found in results/preds")
    print("models:", ", ".join(models))
    main_models = [m for m in models if m in ("denfis_frozen", "xgboost", "lstm")]
    print(plot_equity(main_models))
    print(plot_directional_accuracy(main_models))
    skill_models = [
        m for m in ("no_prediction", "denfis_frozen", "xgboost", "lstm") if m in models
    ]
    if skill_models:
        print(plot_rmse_skill(skill_models))
    if "denfis_online" in models:
        variants = [m for m in ("denfis_frozen", "denfis_online") if m in models]
        print(plot_equity(variants, "denfis_online", online=True))
        print(plot_directional_accuracy(variants, "denfis_online"))


if __name__ == "__main__":
    main()
