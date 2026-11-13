"""Download the external series used by the overnight-feature pilot.

Sources:

* Yahoo Finance chart API for JPM, BAC and XLF (daily adjusted close).
* FRED CSV endpoints for VIXCLS, DEXSIUS (USD/SGD) and DGS10 (10-year yield).

The files are cached under ``data/external/``. The script is idempotent: it
skips a file that already exists unless ``--force`` is passed. Yahoo may rate
limit; the script sleeps between requests.

Run:
    python scripts/download_external.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "external"
START = 1262304000  # 2010-01-01
END = 1794000000    # 2026-11-06
HEADERS = {"User-Agent": "Mozilla/5.0 (research; contact: local)"}
YAHOO = {
    "JPM": "https://query1.finance.yahoo.com/v8/finance/chart/JPM?period1={start}&period2={end}&interval=1d",
    "BAC": "https://query1.finance.yahoo.com/v8/finance/chart/BAC?period1={start}&period2={end}&interval=1d",
    "XLF": "https://query1.finance.yahoo.com/v8/finance/chart/XLF?period1={start}&period2={end}&interval=1d",
}
FRED = {
    "VIXCLS": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS",
    "DEXSIUS": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXSIUS",
    "DGS10": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
}


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _download_yahoo(symbol: str, url: str) -> pd.DataFrame:
    payload = json.loads(_fetch(url))
    result = payload["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adjusted = result["indicators"].get("adjclose", [{}])[0].get("adjclose", quote["close"])
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(timestamps, unit="s", utc=True).date,
            "Close": quote["close"],
            "Adj Close": adjusted,
        }
    ).dropna()
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)


def _fetch_fred(url: str) -> bytes:
    """FRED resets urllib connections here; curl is the reliable fallback."""
    result = subprocess.run(
        ["curl", "-L", "--max-time", "30", "-s", url],
        capture_output=True,
        check=True,
    )
    if not result.stdout:
        raise RuntimeError(f"empty FRED response for {url}")
    return result.stdout


def _download_fred(name: str, url: str) -> pd.DataFrame:
    path = OUT_DIR / f"{name}.csv"
    path.write_bytes(_fetch_fred(url))
    frame = pd.read_csv(path)
    frame.columns = ["Date", name]
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame.dropna().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for symbol, template in YAHOO.items():
        path = OUT_DIR / f"{symbol}.csv"
        if path.exists() and not args.force:
            print(f"skip {path}")
            continue
        frame = _download_yahoo(symbol, template.format(start=START, end=END))
        frame.to_csv(path, index=False)
        print(f"saved {path} ({len(frame)} rows, {frame.Date.min().date()} to {frame.Date.max().date()})")
        time.sleep(2)

    for name, url in FRED.items():
        path = OUT_DIR / f"{name}.csv"
        if path.exists() and not args.force:
            print(f"skip {path}")
            continue
        frame = _download_fred(name, url)
        frame.to_csv(path, index=False)
        print(f"saved {path} ({len(frame)} rows, {frame.Date.min().date()} to {frame.Date.max().date()})")
        time.sleep(1)


if __name__ == "__main__":
    main()
