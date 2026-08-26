"""
utils/dapi_fetch.py
-------------------
Historical funding rates for Binance COIN-margined (inverse) perpetuals from
the dapi endpoint — the data leg of the quanto-convexity study. The inverse
perp's P&L is settled in the ASSET itself (a quanto in Björk ch. 13 terms), so
its fair funding must differ from the linear (USDT-margined) perp's by a
convexity adjustment ~ σ²Δt. Comparing the two funding streams is only
possible with this history.

Mirrors utils/funding_fetch.py exactly (same pagination, same parquet cache
pattern), but against dapi.binance.com and symbols like BTCUSD_PERP. Cache:
data/raw/DFUNDING_{symbol}_{start}_{end}.parquet.
"""

import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

BASE_URL = "https://dapi.binance.com/dapi/v1/fundingRate"
MAX_ROWS = 1000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _rows_to_df(rows: list) -> pd.DataFrame:
    """Raw dapi funding rows → DataFrame (UTC index, float rate). Pure."""
    if not rows:
        return pd.DataFrame(columns=["funding_rate", "mark_price"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["fundingTime"].astype("int64"),
                                     unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df["mark_price"] = pd.to_numeric(df.get("markPrice"), errors="coerce")
    df = df.set_index("timestamp")[["funding_rate", "mark_price"]].sort_index()
    return df[~df.index.duplicated(keep="last")]


def fetch_dapi_funding(symbol: str = "BTCUSD_PERP",
                       start: datetime | None = None,
                       end: datetime | None = None,
                       use_cache: bool = True,
                       request_delay: float = 0.2) -> pd.DataFrame:
    """8-hourly funding history for a coin-margined perp (decimal per 8h)."""
    if start is None:
        start = datetime(2020, 8, 1, tzinfo=timezone.utc)   # coin-M launch era
    if end is None:
        end = datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    cache_file = DATA_RAW_DIR / (
        f"DFUNDING_{symbol.upper()}_"
        f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet")
    if use_cache and cache_file.exists():
        print(f"[dapi_fetch] cache: {cache_file.name}")
        return pd.read_parquet(cache_file)

    print(f"[dapi_fetch] Fetching {symbol} funding {start.date()} → {end.date()} ...")
    all_rows: list = []
    cursor = _to_ms(start)
    end_ms = _to_ms(end)
    while cursor < end_ms:
        resp = requests.get(BASE_URL, params={
            "symbol": symbol.upper(), "startTime": cursor,
            "endTime": end_ms, "limit": MAX_ROWS}, timeout=15)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        cursor = int(rows[-1]["fundingTime"]) + 1
        if len(rows) < MAX_ROWS:
            break
        time.sleep(request_delay)

    df = _rows_to_df(all_rows)
    print(f"[dapi_fetch] Got {len(df):,} funding prints.")
    if use_cache and not df.empty:
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
    return df
