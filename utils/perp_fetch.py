"""
utils/perp_fetch.py
-------------------
Historical PERPETUAL-futures OHLCV from Binance's public USD-M API (fapi),
plus the universe-selection helpers for the cross-sectional book.

WHY A SEPARATE FETCHER. utils/data_fetch.py pulls SPOT klines from
api.binance.com; utils/funding_fetch.py pulls the funding stream. The
cross-sectional momentum strategy actually TRADES the perp (you can short it
and you pay/receive funding on it), so its P&L must be marked on the perp
price, not spot. This mirrors data_fetch.py exactly — same pagination, same
parquet cache pattern — but points at fapi.binance.com/fapi/v1/klines and
caches as PERP_{symbol}_{interval}_{start}_{end}.parquet.

UNIVERSE SELECTION (pre-specified, liquidity × history). select_universe()
ranks the currently-TRADING USDT perps by 24h quote volume and keeps the top
N that were ALSO onboarded on/before a cutoff date — so we get liquid coins
with enough history for a 30-day-lookback weekly strategy. This rule is fixed
BEFORE any P&L is computed. It is still SURVIVORSHIP-TILTED: Binance's API
only lists instruments that exist today, so coins that delisted (the losers)
are absent. Any momentum result on this universe is therefore an OPTIMISTIC
upper bound; the bias is reported, not hidden.
"""

import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
TICKER_24H_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"
MAX_CANDLES_PER_REQUEST = 1500          # fapi klines hard limit (higher than spot's 1000)

INTERVAL_MS = {
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Internal helpers (mirror data_fetch.py)
# ---------------------------------------------------------------------------

def _to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


_KEEP = ["open", "high", "low", "close", "volume", "quote_volume",
         "taker_buy_base_volume", "taker_buy_quote_volume", "trade_count"]


def _raw_to_dataframe(raw: list) -> pd.DataFrame:
    """
    Binance kline raw rows → clean DataFrame (UTC index, floats). Keeps the
    ORDER-FLOW fields too — taker_buy_base/quote_volume and trade_count — so
    aggressive (market-order) buy/sell imbalance is reconstructable from klines
    with full history: taker_sell = volume − taker_buy_base_volume.
    """
    if not raw:
        return pd.DataFrame(columns=_KEEP)
    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trade_count",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"])
    df = df[["timestamp"] + _KEEP].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in _KEEP:
        df[col] = df[col].astype(float)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df[~df.index.duplicated(keep="last")]


def _cache_path(symbol: str, interval: str, start: datetime, end: datetime) -> Path:
    s, e = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    return DATA_RAW_DIR / f"PERP_{symbol.upper()}_{interval}_{s}_{e}.parquet"


# ---------------------------------------------------------------------------
# Public: OHLCV fetch / load
# ---------------------------------------------------------------------------

def fetch_perp_ohlcv(symbol: str, interval: str, start: datetime,
                     end: datetime | None = None, use_cache: bool = True,
                     request_delay: float = 0.2) -> pd.DataFrame:
    """
    Fetch perp OHLCV for a Binance USD-M symbol. Same contract as
    data_fetch.fetch_ohlcv but from fapi and with a quote_volume column.
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unknown interval '{interval}'. Choose {list(INTERVAL_MS)}")
    if end is None:
        end = datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    cache_file = _cache_path(symbol, interval, start, end)
    if use_cache and cache_file.exists():
        print(f"[perp_fetch] cache: {cache_file.name}")
        return pd.read_parquet(cache_file)

    print(f"[perp_fetch] fetch {symbol} {interval} {start.date()}→{end.date()} ...")
    start_ms, end_ms = _to_ms(start), _to_ms(end)
    step_ms = INTERVAL_MS[interval] * MAX_CANDLES_PER_REQUEST
    chunks, cursor = [], start_ms
    while cursor < end_ms:
        chunk_end = min(cursor + step_ms - 1, end_ms)
        resp = requests.get(KLINES_URL, params={
            "symbol": symbol.upper(), "interval": interval,
            "startTime": cursor, "endTime": chunk_end,
            "limit": MAX_CANDLES_PER_REQUEST}, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            break
        chunks.extend(raw)
        cursor = raw[-1][0] + INTERVAL_MS[interval]
        if cursor < end_ms:
            time.sleep(request_delay)

    df = _raw_to_dataframe(chunks)
    df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    print(f"[perp_fetch] {symbol}: {len(df):,} candles")
    if use_cache and not df.empty:
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
    return df


def load_cached_perp(symbol: str, interval: str) -> pd.DataFrame | None:
    """Load the widest-range cached PERP file for a symbol/interval, else None."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = list(DATA_RAW_DIR.glob(f"PERP_{symbol.upper()}_{interval}_*.parquet"))
    if not files:
        return None

    def span(p):
        import re
        m = re.search(r"_(\d{8})_(\d{8})\.parquet$", p.name)
        if not m:
            return -1
        return (pd.Timestamp(m.group(2)) - pd.Timestamp(m.group(1))).days

    return pd.read_parquet(max(files, key=span))


# ---------------------------------------------------------------------------
# Public: universe enumeration & selection
# ---------------------------------------------------------------------------

def list_usdt_perps() -> pd.DataFrame:
    """
    All currently-TRADING USDT perps with their onboard date and 24h quote
    volume. Returns DataFrame indexed by symbol with columns onboard (UTC
    Timestamp) and quote_volume_24h (float).
    """
    info = requests.get(EXCHANGE_INFO_URL, timeout=15).json()["symbols"]
    rows = {s["symbol"]: pd.to_datetime(int(s["onboardDate"]), unit="ms", utc=True)
            for s in info
            if s["quoteAsset"] == "USDT"
            and s.get("contractType") == "PERPETUAL"
            and s["status"] == "TRADING"}
    onboard = pd.Series(rows, name="onboard")

    tick = requests.get(TICKER_24H_URL, timeout=15).json()
    vol = pd.Series({t["symbol"]: float(t["quoteVolume"]) for t in tick},
                    name="quote_volume_24h")

    df = pd.concat([onboard, vol], axis=1).dropna(subset=["onboard"])
    return df.sort_values("quote_volume_24h", ascending=False)


def select_universe(n: int = 25, onboard_before: str = "2022-01-01") -> list[str]:
    """
    Pre-specified selection: the top-`n` USDT perps by 24h quote volume that
    were onboarded on/before `onboard_before`. Liquidity × history, fixed
    before any P&L. SURVIVORSHIP-TILTED (delisted coins are not returned).
    """
    df = list_usdt_perps()
    cutoff = pd.Timestamp(onboard_before, tz="UTC")
    eligible = df[df["onboard"] <= cutoff]
    return list(eligible.head(n).index)
