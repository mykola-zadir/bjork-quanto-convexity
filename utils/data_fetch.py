"""
utils/data_fetch.py
-------------------
Fetches historical OHLCV data from Binance's public REST API.

No API key required — uses unauthenticated public endpoints only.

Key features:
  - Automatic pagination: Binance returns max 1000 candles per request,
    so long date ranges are fetched in chunks and stitched together.
  - Local Parquet cache: once a range is downloaded it is stored in
    data/raw/ so subsequent calls are instant.
  - Strict UTC timestamps throughout — no timezone ambiguity.
  - Returns a clean pandas DataFrame ready for modelling.

Supported timeframes (Binance interval strings):
  1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 1w
"""

import re
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.binance.com/api/v3/klines"
MAX_CANDLES_PER_REQUEST = 1000  # Binance hard limit

# Milliseconds per candle for each interval — used to calculate pagination
INTERVAL_MS = {
    "1m":  60_000,
    "3m":  180_000,
    "5m":  300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h":  3_600_000,
    "2h":  7_200_000,
    "4h":  14_400_000,
    "6h":  21_600_000,
    "8h":  28_800_000,
    "12h": 43_200_000,
    "1d":  86_400_000,
    "1w":  604_800_000,
}

# Project root is one level above this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_ms(dt: datetime) -> int:
    """Convert a UTC-aware (or naive, assumed UTC) datetime to milliseconds."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _fetch_chunk(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """
    Fetch a single chunk of up to 1000 candles from Binance.
    Returns raw list-of-lists from the API.
    """
    params = {
        "symbol":    symbol.upper(),
        "interval":  interval,
        "startTime": start_ms,
        "endTime":   end_ms,
        "limit":     MAX_CANDLES_PER_REQUEST,
    }
    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def _raw_to_dataframe(raw: list) -> pd.DataFrame:
    """
    Convert Binance kline raw response to a clean OHLCV DataFrame.

    Binance kline format (each row):
      [0]  open_time        (ms timestamp)
      [1]  open
      [2]  high
      [3]  low
      [4]  close
      [5]  volume
      [6]  close_time       (ms timestamp)
      [7]  quote_volume
      [8]  trade_count
      [9]  taker_buy_base_volume
      [10] taker_buy_quote_volume
      [11] ignore
    """
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trade_count",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])

    # Keep only what we need
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    # Type conversions
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df


def _cache_path(symbol: str, interval: str, start: datetime, end: datetime) -> Path:
    """Build a deterministic file path for a cached dataset."""
    s = start.strftime("%Y%m%d")
    e = end.strftime("%Y%m%d")
    return DATA_RAW_DIR / f"{symbol.upper()}_{interval}_{s}_{e}.parquet"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime | None = None,
    use_cache: bool = True,
    request_delay: float = 0.25,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a Binance symbol.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. "BTCUSDT", "ETHUSDT"
    interval : str
        Candle size: "1m", "5m", "15m", "1h", "4h", "1d", etc.
    start : datetime
        Start of the requested range (UTC).
    end : datetime, optional
        End of the requested range (UTC). Defaults to now.
    use_cache : bool
        If True, load from local Parquet if it exists; save after fetching.
    request_delay : float
        Seconds to wait between paginated requests (rate-limit courtesy).

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (UTC), columns: open, high, low, close, volume
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unknown interval '{interval}'. Choose from: {list(INTERVAL_MS)}")

    if end is None:
        end = datetime.now(timezone.utc)

    # Normalise to UTC-aware
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    # Check cache
    cache_file = _cache_path(symbol, interval, start, end)
    if use_cache and cache_file.exists():
        print(f"[data_fetch] Loading from cache: {cache_file.name}")
        return pd.read_parquet(cache_file)

    print(f"[data_fetch] Fetching {symbol} {interval} from {start.date()} to {end.date()} ...")

    start_ms = _to_ms(start)
    end_ms   = _to_ms(end)
    step_ms  = INTERVAL_MS[interval] * MAX_CANDLES_PER_REQUEST

    all_chunks = []
    cursor = start_ms

    while cursor < end_ms:
        chunk_end = min(cursor + step_ms - 1, end_ms)
        raw = _fetch_chunk(symbol, interval, cursor, chunk_end)

        if not raw:
            break

        all_chunks.extend(raw)

        # Advance cursor past the last returned candle
        last_open_time = raw[-1][0]
        cursor = last_open_time + INTERVAL_MS[interval]

        if cursor < end_ms:
            time.sleep(request_delay)  # be polite to the API

    df = _raw_to_dataframe(all_chunks)

    # Trim to exactly the requested range
    df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

    print(f"[data_fetch] Got {len(df):,} candles.")

    # Save to cache
    if use_cache and not df.empty:
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
        print(f"[data_fetch] Saved to cache: {cache_file.name}")

    return df


def _cache_span(path: Path) -> tuple[int, str]:
    """
    (span_in_days, end_str) parsed from a cache filename of the form
    SYMBOL_interval_YYYYMMDD_YYYYMMDD.parquet. Unparseable names sort last.
    Used to choose the cache covering the WIDEST date range.
    """
    m = re.search(r"_(\d{8})_(\d{8})\.parquet$", path.name)
    if not m:
        return (-1, "")
    start, end = m.group(1), m.group(2)
    return (int((pd.Timestamp(end) - pd.Timestamp(start)).days), end)


def widest_cache(files: list[Path]) -> Path:
    """
    Pick the cache file spanning the widest date range (tie-break: latest end).

    Sorting filenames lexicographically is WRONG: a short cache whose START
    date is later (e.g. BTCUSDT_1h_20260512_20260519) sorts after the full
    history (BTCUSDT_1h_20210101_20260612) and would be silently returned,
    handing a study a few days of data instead of years.
    """
    return max(files, key=_cache_span)


def load_cached(symbol: str, interval: str) -> pd.DataFrame | None:
    """
    Load the cached file with the WIDEST date range for a symbol/interval.
    Returns None if nothing is cached. (Widest, not lexicographically last —
    see widest_cache: the old "files[-1]" could return a tiny recent slice.)
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    pattern = f"{symbol.upper()}_{interval}_*.parquet"
    files = list(DATA_RAW_DIR.glob(pattern))
    if not files:
        return None
    latest = widest_cache(files)
    print(f"[data_fetch] Loading widest cache: {latest.name}")
    return pd.read_parquet(latest)
