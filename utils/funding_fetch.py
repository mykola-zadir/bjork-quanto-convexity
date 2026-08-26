"""
utils/funding_fetch.py
----------------------
Historical perpetual-futures FUNDING RATES from Binance's public API.

WHAT FUNDING IS. A perpetual future has no expiry, so an 8-hourly cash flow
keeps it pinned to spot: when the perp trades ABOVE the index (leveraged
longs crowding in), longs pay shorts a funding rate; below, shorts pay
longs. The rate is therefore a direct, observable gauge of POSITIONING —
how crowded and how levered the speculative side of the market is right now.

WHY THIS PROJECT WANTS IT. Crowded leverage is liquidation fuel: when price
moves against a crowded side, forced liquidations beget further moves — the
cascade mechanism behind sudden vol explosions (Aug-2024 style gaps). Our
MS-GARCH gate is RETURN-driven: it cannot flag a storm until the first
storm-sized return has already hit the book. Funding is FORWARD-LOOKING
positioning data, available before the gap. Whether it actually carries
incremental information about next-day volatility is exactly what
verify_funding_btc.py tests (pre-registered, F1/F2).

API: GET fapi.binance.com/fapi/v1/fundingRate — public, no key, max 1000
rows per request (8h prints → ~333 days/page). Same parquet cache pattern
as the other fetchers.
"""

import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

BASE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
MAX_ROWS = 1000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _rows_to_df(rows: list) -> pd.DataFrame:
    """Raw funding rows → DataFrame (UTC index, float rate). Pure function."""
    if not rows:
        return pd.DataFrame(columns=["funding_rate", "mark_price"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["fundingTime"].astype("int64"),
                                     unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df["mark_price"] = pd.to_numeric(df.get("markPrice"), errors="coerce")
    df = df.set_index("timestamp")[["funding_rate", "mark_price"]].sort_index()
    return df[~df.index.duplicated(keep="last")]


def fetch_funding(
    symbol: str = "BTCUSDT",
    start: datetime | None = None,
    end: datetime | None = None,
    use_cache: bool = True,
    request_delay: float = 0.2,
) -> pd.DataFrame:
    """
    Fetch the 8-hourly funding-rate history for a Binance perp.

    Returns a DataFrame indexed by funding print time (UTC) with columns
    funding_rate (decimal per 8h, e.g. 0.0001 = 1bp) and mark_price.
    """
    if start is None:
        start = datetime(2021, 1, 1, tzinfo=timezone.utc)
    if end is None:
        end = datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    cache_file = DATA_RAW_DIR / (
        f"FUNDING_{symbol.upper()}_"
        f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet")
    if use_cache and cache_file.exists():
        print(f"[funding_fetch] Loading from cache: {cache_file.name}")
        return pd.read_parquet(cache_file)

    print(f"[funding_fetch] Fetching {symbol} funding "
          f"{start.date()} → {end.date()} ...")
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
    print(f"[funding_fetch] Got {len(df):,} funding prints.")
    if use_cache and not df.empty:
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
        print(f"[funding_fetch] Saved to cache: {cache_file.name}")
    return df


def daily_funding(funding: pd.DataFrame) -> pd.Series:
    """
    Sum the day's 8-hourly prints into the day's total funding rate.
    The prints at 00:00/08:00/16:00 UTC are all known before the daily
    close, so day t's value is usable in a decision made at day t's close.
    """
    return funding["funding_rate"].groupby(pd.Grouper(freq="1D")).sum()


def crowding_gauge(fund_d: pd.Series, smooth: int = 3, burn_in: int = 60,
                   window: int | None = None) -> pd.Series:
    """
    Causal positioning gauge: the percentile of today's |smooth-day mean
    daily funding| within its own history.

    window=None  → EXPANDING history (all past days). Caveat learned the
        hard way: when funding levels shift regime (2021's wild prints vs
        the compressed years after), the expanding gauge is forever anchored
        to the old regime and stops firing — same failure mode as comparing
        DVOL to its all-time history.
    window=365   → ROLLING history: crowding relative to the CURRENT regime.

    Both are strictly causal: day t's percentile uses days < t only.
    """
    f = fund_d.rolling(smooth, min_periods=smooth).mean().abs()
    vals = f.values
    g = pd.Series(np.nan, index=f.index)
    for i in range(burn_in, len(vals)):
        if not np.isfinite(vals[i]):
            continue
        lo = 0 if window is None else max(0, i - window)
        past = vals[lo:i]
        past = past[np.isfinite(past)]
        if len(past):
            g.iloc[i] = (past < vals[i]).mean()
    return g
