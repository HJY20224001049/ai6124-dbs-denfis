"""Data download, cleaning, caching, and loading.

Primary source is the Yahoo Finance chart endpoint, which returns the same
daily OHLCV and adjusted-close series as yfinance but without the library's
rate-limit wrapper.  The final notebook only reads the cached CSVs written by
this module; it never needs network access.

Protocol notes
--------------
* D05.SI: rows with missing OHLC/adj-close or Volume <= 0 are dropped.
* ^STI is an index: its volume is frequently zero, so the volume filter does
  not apply.  Missing volume is stored as 0 and price rows are kept.  The
  transfer-learning report documents that index volume is not informative.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

try:  # allow `python src/data.py` as well as `python -m src.data`
    from config import (
        DATA_DIR,
        DATE_COL,
        END_DATE,
        PRICE_COLS,
        START_DATE,
        SYMBOL_DBS,
        SYMBOL_STI,
    )
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import (  # type: ignore
        DATA_DIR,
        DATE_COL,
        END_DATE,
        PRICE_COLS,
        START_DATE,
        SYMBOL_DBS,
        SYMBOL_STI,
    )

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}
SG_TZ = ZoneInfo("Asia/Singapore")


def _epoch(date_str: str) -> int:
    """UTC midnight for a YYYY-MM-DD date, as a Unix timestamp."""
    return int(pd.Timestamp(date_str, tz="UTC").timestamp())


def _to_sg_date(ts: int) -> str:
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .astimezone(SG_TZ)
        .date()
        .isoformat()
    )


def download_prices(
    symbol: str,
    start: str = START_DATE,
    end: str = END_DATE,
    retries: int = 4,
    pause: float = 3.0,
) -> pd.DataFrame:
    """Download daily bars for one symbol and return a raw DataFrame.

    `end` is inclusive: the request asks for one extra day so that a bar
    stamped at the end date is returned.
    """
    params = {
        "period1": _epoch(start),
        "period2": _epoch(end) + 86400,
        "interval": "1d",
        "events": "div,splits",
    }
    url = CHART_URL.format(symbol=requests.utils.quote(symbol, safe=""))

    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if response.status_code == 429:
                raise RuntimeError("Yahoo returned 429 (rate limited)")
            response.raise_for_status()
            payload = response.json()
            chart = payload.get("chart") or {}
            if chart.get("error"):
                raise RuntimeError(f"Yahoo chart error: {chart['error']}")
            result = chart["result"][0]
            break
        except Exception as exc:  # noqa: BLE001 - retry any transient failure
            last_error = exc
            if attempt == retries - 1:
                raise RuntimeError(
                    f"Failed to download {symbol} after {retries} attempts: {exc}"
                ) from exc
            time.sleep(pause * (2**attempt))
    else:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to download {symbol}: {last_error}")

    timestamps = result.get("timestamp") or []
    quote = result["indicators"]["quote"][0]
    adjclose_block = result["indicators"].get("adjclose")
    adjclose = (
        adjclose_block[0]["adjclose"]
        if adjclose_block
        else quote.get("close")
    )

    if len(timestamps) != len(quote["close"]) or len(timestamps) != len(adjclose):
        raise ValueError(f"Inconsistent arrays for {symbol}")

    df = pd.DataFrame(
        {
            DATE_COL: [_to_sg_date(ts) for ts in timestamps],
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Adj Close": adjclose,
            "Volume": quote.get("volume"),
        }
    )
    return df


def clean_prices(df: pd.DataFrame, is_index: bool = False) -> pd.DataFrame:
    """Apply the protocol's cleaning rules and return a sorted DataFrame."""
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL]).dt.date
    out = out.sort_values(DATE_COL).drop_duplicates(DATE_COL, keep="last")

    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce")

    out = out.replace([float("inf"), float("-inf")], pd.NA)
    out = out.dropna(subset=["Open", "High", "Low", "Close", "Adj Close"])

    if is_index:
        out["Volume"] = out["Volume"].fillna(0.0)
    else:
        out = out[out["Volume"].fillna(0.0) > 0]

    return out.reset_index(drop=True)


def save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, date_format="%Y-%m-%d")


def load_prices(name: str) -> pd.DataFrame:
    """Load a cached file by name: 'D05_SI' or 'STI'."""
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python -m src.data` to download and cache data."
        )
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df[DATE_COL] = df[DATE_COL].dt.date
    return df


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(paths: list[Path], out_path: Path) -> dict[str, str]:
    checksums = {p.name: sha256_file(p) for p in paths}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for name, digest in checksums.items():
            handle.write(f"{digest}  {name}\n")
    return checksums


def download_all() -> dict:
    """Download both series, clean, cache, and write checksums + metadata."""
    specs = [
        ("D05_SI", SYMBOL_DBS, False),
        ("STI", SYMBOL_STI, True),
    ]
    summary: dict[str, dict] = {}
    written: list[Path] = []

    for name, symbol, is_index in specs:
        raw = download_prices(symbol)
        clean = clean_prices(raw, is_index=is_index)
        path = DATA_DIR / f"{name}.csv"
        save_cache(clean, path)
        written.append(path)
        summary[name] = {
            "symbol": symbol,
            "rows": int(len(clean)),
            "first_date": str(clean[DATE_COL].iloc[0]),
            "last_date": str(clean[DATE_COL].iloc[-1]),
            "is_index": is_index,
        }
        print(
            f"{name:7s} {symbol:8s} rows={len(clean):5d}  "
            f"{clean[DATE_COL].iloc[0]} .. {clean[DATE_COL].iloc[-1]}"
        )

    checksums = write_checksums(written, DATA_DIR / "checksums.txt")
    metadata = {
        "source": "Yahoo Finance chart API",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "start": START_DATE,
        "end": END_DATE,
        "series": summary,
        "sha256": checksums,
    }
    meta_path = DATA_DIR / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(f"Wrote {meta_path}")
    return metadata


if __name__ == "__main__":
    download_all()
