"""Step 2: cut the full inventory down to what a small account can actually trade.

With a fixed budget the enemy is not "finding the best coin", it is slippage
and spread on illiquid pairs. Everything here is a conservative filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class UniverseConfig:
    min_turnover_usd: float = 2_000_000  # 24h turnover
    max_spread_bps: float = 15.0
    min_oi_usd: float = 1_000_000  # perps only
    quote_coins: tuple[str, ...] = ("USDT", "USDC")
    exclude_innovation: bool = True
    exclude_dated_futures: bool = True
    # options are inventoried but never auto-traded: too illiquid for small size
    include_options: bool = False


def filter_universe(inv: pd.DataFrame, cfg: UniverseConfig = UniverseConfig()) -> pd.DataFrame:
    df = inv.copy()
    df = df[df.status == "Trading"]
    df = df[~df.isPreListing.astype(bool)]
    if not cfg.include_options:
        df = df[df.category != "option"]
    if cfg.exclude_innovation and "innovation" in df:
        df = df[df.innovation.fillna("0").astype(str) != "1"]
    if cfg.exclude_dated_futures:
        df = df[~df.contractType.fillna("").str.contains("Futures", case=False)]

    quote_ok = df.quoteCoin.isin(cfg.quote_coins) | (df.category == "inverse")
    df = df[quote_ok]
    df = df[df.turnover24h >= cfg.min_turnover_usd]
    df = df[df.spread_bps.fillna(1e9) <= cfg.max_spread_bps]

    is_perp = df.category.isin(["linear", "inverse"])
    df = df[~is_perp | (df.openInterestValue.fillna(0) >= cfg.min_oi_usd)]

    return df.sort_values("turnover24h", ascending=False).reset_index(drop=True)
