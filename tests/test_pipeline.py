"""Offline tests: a fake Bybit client with synthetic markets drives the whole pipeline."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

from bybit_scan import backtest as bt
from bybit_scan.client import BybitPublic
from bybit_scan.data import fetch_klines, klines_df
from bybit_scan.features import build_feature_table, instrument_features
from bybit_scan.inventory import fetch_inventory, summarize_inventory
from bybit_scan.report import build_report
from bybit_scan.universe import UniverseConfig, filter_universe

DAY = 86_400_000


def synth_prices(n: int, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "trend":
        r = rng.normal(0.003, 0.02, n)
    elif kind == "meanrev":
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.8 * x[i - 1] + rng.normal(0, 0.03)
        return 100 * np.exp(x)
    else:
        r = rng.normal(0, 0.03, n)
    return 100 * np.exp(np.cumsum(r))


class FakeResponse:
    def __init__(self, payload, status=200):
        self.status_code, self._p, self.text = status, payload, json.dumps(payload)

    def json(self):
        return self._p

    def raise_for_status(self):
        pass


class FakeSession:
    """Serves an instruments/tickers/kline/funding API for three synthetic markets."""

    markets = {
        ("spot", "TRENDUSDT"): "trend",
        ("linear", "TRENDUSDT"): "trend",
        ("linear", "MEANUSDT"): "meanrev",
        ("spot", "ILLIQUSDT"): "noise",
        ("option", "BTC-31DEC26-100000-C"): None,
    }

    def __init__(self):
        self.calls = 0
        self.now = int(time.time() * 1000) // DAY * DAY

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        p = params or {}
        cat = p.get("category")
        if url.endswith("/instruments-info"):
            lst = []
            for (c, s), _ in self.markets.items():
                if c != cat:
                    continue
                lst.append({"symbol": s, "status": "Trading", "baseCoin": s.split("USDT")[0] if "USDT" in s else "BTC",
                            "quoteCoin": "USDT" if "USDT" in s else "USDC", "contractType": "LinearPerpetual" if c == "linear" else None,
                            "lotSizeFilter": {"minOrderQty": "0.001"}, "priceFilter": {"tickSize": "0.01"},
                            "leverageFilter": {"maxLeverage": "50"}})
            if p.get("status") == "PreLaunch":
                lst = []
            return FakeResponse({"retCode": 0, "result": {"list": lst, "nextPageCursor": ""}})
        if url.endswith("/tickers"):
            lst = []
            for (c, s), kind in self.markets.items():
                if c != cat:
                    continue
                illiquid = "ILLIQ" in s
                lst.append({"symbol": s, "lastPrice": "100", "bid1Price": "99.95" if not illiquid else "95",
                            "ask1Price": "100.05" if not illiquid else "105", "turnover24h": "50000000" if not illiquid else "1000",
                            "volume24h": "1", "openInterestValue": "20000000" if c == "linear" else None,
                            "fundingRate": "0.0003" if c == "linear" else None})
            return FakeResponse({"retCode": 0, "result": {"list": lst}})
        if url.endswith("/kline"):
            kind = self.markets[(cat, p["symbol"])]
            n = 400
            px = synth_prices(n, kind, seed=hash(p["symbol"]) % 1000)
            end = min(int(p.get("end") or self.now), self.now)
            rows = []
            for i in range(n):
                ts = self.now - (n - 1 - i) * DAY
                if ts <= end and ts >= int(p.get("start") or 0):
                    rows.append([str(ts), str(px[i]), str(px[i] * 1.01), str(px[i] * 0.99), str(px[i]), "1000", str(px[i] * 1000)])
            rows = rows[-int(p.get("limit", 1000)):]
            return FakeResponse({"retCode": 0, "result": {"list": list(reversed(rows))}})
        if url.endswith("/funding/history"):
            end = int(p.get("endTime") or self.now)
            rows = [{"symbol": p["symbol"], "fundingRate": "0.0003", "fundingRateTimestamp": str(end - i * DAY // 3)} for i in range(150)]
            return FakeResponse({"retCode": 0, "result": {"list": rows}})
        if url.endswith("/time"):
            return FakeResponse({"retCode": 0, "result": {"timeSecond": str(self.now // 1000)}})
        raise AssertionError(url)


@pytest.fixture
def client():
    return BybitPublic(cache_dir=None, rate=10_000, session=FakeSession())


def test_inventory_covers_all_categories(client):
    inv = fetch_inventory(client)
    assert set(inv.category) == {"spot", "linear", "option"}
    assert len(inv) == 5
    assert inv.loc[inv.symbol == "ILLIQUSDT", "spread_bps"].iloc[0] > 500
    summ = summarize_inventory(inv)
    assert summ.loc["spot", "instruments"] == 2


def test_universe_filter_drops_illiquid_and_options(client):
    inv = fetch_inventory(client)
    uni = filter_universe(inv, UniverseConfig())
    assert set(uni.symbol) == {"TRENDUSDT", "MEANUSDT"}
    assert "option" not in set(uni.category)


def test_kline_paging_and_df(client):
    df = fetch_klines(client, "spot", "TRENDUSDT", "D", days=365)
    assert 300 <= len(df) <= 366
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()
    assert klines_df([]).empty


def test_backtest_no_lookahead_and_costs():
    idx = pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC")
    close = pd.Series(synth_prices(300, "trend", 1), index=idx)
    # position decided on close t is applied to the return of t+1
    pos = pd.Series(0.0, index=idx)
    pos.iloc[10] = 1.0
    r = bt.run("x", close, pos, bt.Costs(0.001, 0.0))
    expected = close.iloc[11] / close.iloc[10] - 1
    got = r.equity.iloc[10] / r.equity.iloc[9] - 1  # equity index starts at bar 1
    assert got == pytest.approx(expected - 0.001)  # entry cost paid on bar 10 → shows at idx 9 in ret
    assert r.trades == 2
    bh = bt.run("bh", close, bt.buy_and_hold(close), bt.Costs(0.0, 0.0))
    assert bh.total_return == pytest.approx(close.iloc[-1] / close.iloc[0] - 1, rel=1e-9)


def test_strategies_behave_as_designed():
    idx = pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC")
    trend = pd.Series(synth_prices(400, "trend", 2), index=idx)
    mr = pd.Series(synth_prices(400, "meanrev", 3), index=idx)
    assert bt.run("t", trend, bt.sma_trend(trend), bt.SPOT_COSTS).total_return > 0.5
    assert bt.run("m", mr, bt.mean_reversion(mr), bt.PERP_COSTS).sharpe > 0.5
    vt = bt.vol_target(bt.sma_trend(trend), trend, 0.1)
    assert vt.max() <= 1.0 and vt.min() >= 0.0


def test_funding_carry_is_funding_minus_costs():
    idx = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    close = pd.Series(100.0, index=idx)
    f_idx = pd.date_range("2024-01-01", periods=300, freq="8h", tz="UTC")
    funding = pd.Series(0.0001, index=f_idx)
    r = bt.funding_carry(close, funding, bt.PERP_COSTS)
    gross = 0.0001 * 300
    costs = 2 * (bt.PERP_COSTS.fee + bt.PERP_COSTS.slippage + bt.SPOT_COSTS.fee + bt.SPOT_COSTS.slippage)
    # returns compound daily, so allow a little drift above the arithmetic sum
    assert r.total_return == pytest.approx(gross - costs, abs=6e-4)
    assert r.trades == 2 and r.extra["days"] == 100


def test_features_and_report_end_to_end(client):
    inv = fetch_inventory(client)
    uni = filter_universe(inv)
    from bybit_scan.data import fetch_all_funding, fetch_history

    hist = fetch_history(client, uni, "D", 400, workers=2, progress=lambda *_: None)
    fund = fetch_all_funding(client, uni, 100, workers=2)
    feats = build_feature_table(hist, fund)
    assert feats.loc[("spot", "TRENDUSDT"), "hurst"] > feats.loc[("linear", "MEANUSDT"), "hurst"]
    assert feats.loc[("linear", "MEANUSDT"), "funding_ann"] == pytest.approx(0.0003 * 3 * 365)
    res = bt.evaluate_universe(hist, fund)
    assert "funding_carry_neutral" in set(res.strategy)
    md = build_report(summarize_inventory(inv), uni, feats, res, budget=1000)
    assert "## 4. Backtests" in md and "TRENDUSDT" in md


def test_short_history_features():
    df = pd.DataFrame({"close": [1.0] * 10})
    assert instrument_features(df) == {"bars": 10}
