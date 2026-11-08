"""Exploratory data analysis for the cached D05.SI and ^STI series.

Run from the project root:
    python scripts/eda.py
Figures are written to docs/figs/ and a summary to docs/eda_summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATE_COL, FIGURES_DIR  # noqa: E402
from src.data import load_prices  # noqa: E402

OUT_DIR = ROOT / "docs" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.autolayout": True,
    }
)

INK = "#16211E"
ACCENT = "#0E6B5A"
WARN = "#94510F"
MUTED = "#5A6964"


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def main() -> None:
    dbs = load_prices("D05_SI")
    sti = load_prices("STI")

    for frame in (dbs, sti):
        frame[DATE_COL] = pd.to_datetime(frame[DATE_COL])

    dbs["log_ret"] = log_returns(dbs["Close"])
    sti["log_ret"] = log_returns(sti["Close"])

    # ------------------------------------------------------------------
    # 1. Price and volume
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, height_ratios=[3, 1])
    axes[0].plot(dbs[DATE_COL], dbs["Close"], color=INK, linewidth=0.9)
    axes[0].set_title("DBS Group Holdings (D05.SI) — adjusted close and volume")
    axes[0].set_ylabel("Price (SGD)")
    axes[1].bar(dbs[DATE_COL], dbs["Volume"] / 1e6, color=ACCENT, width=1.0, alpha=0.55)
    axes[1].set_ylabel("Volume (million)")
    axes[1].set_xlabel("Date")
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    fig.savefig(OUT_DIR / "01_price_volume.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Return distribution and cumulative return
    # ------------------------------------------------------------------
    rets = dbs["log_ret"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(rets, bins=80, color=ACCENT, alpha=0.75, edgecolor="white", linewidth=0.2)
    axes[0].axvline(0, color=MUTED, linewidth=0.8)
    axes[0].set_title("Daily log-return distribution")
    axes[0].set_xlabel("Log return")
    axes[0].set_ylabel("Count")

    cum = np.exp(rets.cumsum()) - 1
    axes[1].plot(dbs[DATE_COL].iloc[1:], cum, color=INK, linewidth=1.0)
    axes[1].set_title("Cumulative simple return")
    axes[1].set_ylabel("Return")
    axes[1].xaxis.set_major_locator(mdates.YearLocator(4))
    fig.savefig(OUT_DIR / "02_returns.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Data coverage per year
    # ------------------------------------------------------------------
    dbs["year"] = dbs[DATE_COL].dt.year
    sti["year"] = sti[DATE_COL].dt.year
    counts = (
        dbs.groupby("year").size().rename("D05.SI").to_frame()
        .join(sti.groupby("year").size().rename("^STI"), how="outer")
        .fillna(0)
        .astype(int)
    )
    fig, ax = plt.subplots(figsize=(10, 3.2))
    x = np.arange(len(counts))
    width = 0.4
    ax.bar(x - width / 2, counts["D05.SI"], width, label="D05.SI", color=ACCENT)
    ax.bar(x + width / 2, counts["^STI"], width, label="^STI", color=WARN, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, rotation=45, ha="right")
    ax.set_ylabel("Trading rows")
    ax.set_title("Rows per calendar year after cleaning")
    ax.legend(frameon=False)
    fig.savefig(OUT_DIR / "03_data_coverage.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Return anomalies
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.plot(dbs[DATE_COL], dbs["log_ret"], color=MUTED, linewidth=0.6)
    big = dbs[dbs["log_ret"].abs() > 0.05]
    ax.scatter(big[DATE_COL], big["log_ret"], color=WARN, s=12, zorder=3, label="|return| > 5%")
    ax.axhline(0, color=INK, linewidth=0.6)
    ax.set_title("Daily log returns and large moves")
    ax.set_ylabel("Log return")
    ax.legend(frameon=False, loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.savefig(OUT_DIR / "04_return_anomalies.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {
        "D05_SI": {
            "rows": int(len(dbs)),
            "first": str(dbs[DATE_COL].iloc[0].date()),
            "last": str(dbs[DATE_COL].iloc[-1].date()),
            "start_close": float(dbs["Close"].iloc[0]),
            "end_close": float(dbs["Close"].iloc[-1]),
            "max_abs_return": float(rets.abs().max()),
            "max_abs_return_date": str(dbs.loc[rets.abs().idxmax(), DATE_COL].date()),
            "positive_days_pct": float((rets > 0).mean() * 100),
            "annualized_vol_pct": float(rets.std() * np.sqrt(252) * 100),
            "zero_volume_rows": int((dbs["Volume"] <= 0).sum()),
        },
        "^STI": {
            "rows": int(len(sti)),
            "first": str(sti[DATE_COL].iloc[0].date()),
            "last": str(sti[DATE_COL].iloc[-1].date()),
            "zero_volume_rows": int((sti["Volume"] <= 0).sum()),
        },
    }
    (ROOT / "docs" / "eda_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
