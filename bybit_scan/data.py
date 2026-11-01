"""Step 3: pull OHLCV history and funding for the tradeable universe."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .client import BybitPublic

INTERVAL_MS = {"60": 3_600_000, "240": 14_400_000, "D": 86_400_000}


def klines_df(raw: list[list[str]]) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "turnover"])
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume", "turnover"]).astype(float)
    df["ts"] = pd.to_datetime(df.ts, unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    return df[~df.index.duplicated(keep="last")]


def fetch_klines(client: BybitPublic, category: str, symbol: str, interval: str = "D", days: int = 730) -> pd.DataFrame:
    """Fetch up to `days` of candles, paging backwards 1000 at a time."""
    step = INTERVAL_MS[interval]
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    frames: list[pd.DataFrame] = []
    cursor_end = end
    while cursor_end > start:
        raw = client.kline(category, symbol, interval, start=start, end=cursor_end, limit=1000)
        if not raw:
            break
        df = klines_df(raw)
        frames.append(df)
        oldest = int(df.index[0].timestamp() * 1000)
        if len(raw) < 1000 or oldest <= start:
            break
        cursor_end = oldest - step
    if not frames:
        return klines_df([])
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


def fetch_funding(client: BybitPublic, category: str, symbol: str, days: int = 365) -> pd.Series:
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    rows: list[dict] = []
    cursor_end = end
    while cursor_end > start:
        batch = client.funding_history(category, symbol, start=start, end=cursor_end)
        if not batch:
            break
        rows += batch
        oldest = min(int(b["fundingRateTimestamp"]) for b in batch)
        if len(batch) < 200 or oldest <= start:
            break
        cursor_end = oldest - 1
    if not rows:
        return pd.Series(dtype=float, name="fundingRate")
    s = pd.Series(
        [float(r["fundingRate"]) for r in rows],
        index=pd.to_datetime([int(r["fundingRateTimestamp"]) for r in rows], unit="ms", utc=True),
        name="fundingRate",
    ).sort_index()
    return s[~s.index.duplicated(keep="last")]


def fetch_history(
    client: BybitPublic,
    universe: pd.DataFrame,
    interval: str = "D",
    days: int = 730,
    workers: int = 6,
    progress=print,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Download candles for every row of `universe`. Returns {(category, symbol): df}."""
    keys = list(zip(universe.category, universe.symbol))
    out: dict[tuple[str, str], pd.DataFrame] = {}
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_klines, client, c, s, interval, days): (c, s) for c, s in keys}
        for i, f in enumerate(as_completed(futs), 1):
            key = futs[f]
            try:
                out[key] = f.result()
            except Exception as e:  # noqa: BLE001
                progress(f"  ! {key}: {e}")
            if i % 25 == 0 or i == len(futs):
                progress(f"  klines {i}/{len(futs)}")
    return out


def fetch_all_funding(client: BybitPublic, universe: pd.DataFrame, days: int = 365, workers: int = 6) -> dict[str, pd.Series]:
    perps = universe[universe.category.isin(["linear", "inverse"])]
    out: dict[str, pd.Series] = {}
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_funding, client, c, s, days): s for c, s in zip(perps.category, perps.symbol)}
        for f in as_completed(futs):
            try:
                out[futs[f]] = f.result()
            except Exception:  # noqa: BLE001
                pass
    return out
