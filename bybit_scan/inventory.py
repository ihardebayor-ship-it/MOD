"""Step 1: inventory every instrument Bybit exposes through the V5 API.

Covers spot (including tokenized stocks / RWA pairs, which trade as spot),
USDT/USDC perpetuals and dated futures (linear), inverse contracts, pre-market
contracts, and options. Joined with 24h tickers so each row carries turnover,
spread, open interest and funding rate.
"""

from __future__ import annotations

import pandas as pd

from .client import CATEGORIES, BybitPublic


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _flatten_instrument(cat: str, it: dict) -> dict:
    lot = it.get("lotSizeFilter", {}) or {}
    pf = it.get("priceFilter", {}) or {}
    lev = it.get("leverageFilter", {}) or {}
    return {
        "category": cat,
        "symbol": it.get("symbol"),
        "status": it.get("status"),
        "baseCoin": it.get("baseCoin"),
        "quoteCoin": it.get("quoteCoin"),
        "settleCoin": it.get("settleCoin"),
        "contractType": it.get("contractType"),
        "symbolType": it.get("symbolType"),
        "optionsType": it.get("optionsType"),
        "launchTime": _f(it.get("launchTime")),
        "deliveryTime": _f(it.get("deliveryTime")),
        "tickSize": _f(pf.get("tickSize")),
        "minOrderQty": _f(lot.get("minOrderQty")),
        "minOrderAmt": _f(lot.get("minOrderAmt") or lot.get("minNotionalValue")),
        "maxLeverage": _f(lev.get("maxLeverage")),
        "innovation": it.get("innovation"),
        "isPreListing": bool(it.get("isPreListing", False)),
    }


def _flatten_ticker(t: dict) -> dict:
    bid, ask, last = _f(t.get("bid1Price")), _f(t.get("ask1Price")), _f(t.get("lastPrice"))
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
    spread_bps = (ask - bid) / mid * 1e4 if bid > 0 and ask > 0 and mid > 0 else float("nan")
    return {
        "symbol": t.get("symbol"),
        "lastPrice": last,
        "price24hPcnt": _f(t.get("price24hPcnt")),
        "turnover24h": _f(t.get("turnover24h")),
        "volume24h": _f(t.get("volume24h")),
        "openInterestValue": _f(t.get("openInterestValue")),
        "fundingRate": _f(t.get("fundingRate")),
        "spread_bps": spread_bps,
        "markIv": _f(t.get("markIv")),
        "delta": _f(t.get("delta")),
    }


def fetch_inventory(client: BybitPublic, categories=CATEGORIES, include_premarket: bool = True) -> pd.DataFrame:
    rows: list[dict] = []
    for cat in categories:
        seen: set[str] = set()
        statuses: list[str | None] = [None]
        if include_premarket and cat in ("linear", "inverse"):
            statuses.append("PreLaunch")
        for status in statuses:
            for it in client.instruments(cat, status=status):
                if it["symbol"] in seen:
                    continue
                seen.add(it["symbol"])
                rows.append(_flatten_instrument(cat, it))
    inst = pd.DataFrame(rows)

    tick_rows: list[dict] = []
    for cat in categories:
        if cat == "option":
            bases = sorted(inst.loc[inst.category == "option", "baseCoin"].dropna().unique())
            for b in bases:
                tick_rows += [{"category": cat, **_flatten_ticker(t)} for t in client.tickers(cat, base_coin=b)]
        else:
            tick_rows += [{"category": cat, **_flatten_ticker(t)} for t in client.tickers(cat)]
    ticks = pd.DataFrame(tick_rows)
    if ticks.empty:
        return inst
    return inst.merge(ticks, on=["category", "symbol"], how="left")


def summarize_inventory(inv: pd.DataFrame) -> pd.DataFrame:
    g = inv.groupby("category")
    out = pd.DataFrame(
        {
            "instruments": g.size(),
            "trading": g.apply(lambda d: int((d.status == "Trading").sum())),
            "turnover24h_usd": g.turnover24h.sum(),
            "median_spread_bps": g.spread_bps.median(),
        }
    )
    return out.sort_values("turnover24h_usd", ascending=False)
