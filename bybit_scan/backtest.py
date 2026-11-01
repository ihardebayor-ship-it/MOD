"""Step 5: vectorised daily backtests with fees, slippage and an out-of-sample split.

Strategies are deliberately simple. If a simple rule does not survive fees
on out-of-sample data, a complicated one usually will not either.

Positions are decided on bar t's close and held through bar t+1, so there is
no look-ahead. Every change in position pays `fee + slippage` on the traded
notional.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ANN = 365


@dataclass
class Costs:
    # Bybit defaults (taker). Spot: 0.10%. Perps: 0.055%. Adjust for VIP tiers.
    fee: float = 0.001
    slippage: float = 0.0005


SPOT_COSTS = Costs(fee=0.0010, slippage=0.0005)
PERP_COSTS = Costs(fee=0.00055, slippage=0.0005)


@dataclass
class Result:
    name: str
    equity: pd.Series
    position: pd.Series
    total_return: float
    cagr: float
    sharpe: float
    max_dd: float
    trades: int
    exposure: float
    oos_return: float = float("nan")
    oos_sharpe: float = float("nan")
    extra: dict = field(default_factory=dict)

    def row(self) -> dict:
        return {
            "strategy": self.name,
            "total_return": self.total_return,
            "cagr": self.cagr,
            "sharpe": self.sharpe,
            "max_dd": self.max_dd,
            "trades": self.trades,
            "exposure": self.exposure,
            "oos_return": self.oos_return,
            "oos_sharpe": self.oos_sharpe,
            **self.extra,
        }


def _metrics(ret: pd.Series) -> tuple[float, float, float, float]:
    eq = (1 + ret).cumprod()
    total = float(eq.iloc[-1] - 1) if len(eq) else 0.0
    years = max(len(ret) / ANN, 1e-9)
    cagr = float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0
    sharpe = float(ret.mean() / ret.std() * np.sqrt(ANN)) if ret.std() > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0
    return total, cagr, sharpe, mdd


def run(name: str, close: pd.Series, position: pd.Series, costs: Costs, oos_frac: float = 0.3, carry: pd.Series | None = None) -> Result:
    """position: target exposure in [-1, 1] decided at each bar's close."""
    close = close.dropna()
    position = position.reindex(close.index).fillna(0.0).clip(-1, 1)
    bar_ret = close.pct_change().fillna(0.0)
    held = position.shift(1).fillna(0.0)  # what we held going into this bar
    turnover = (position - held).abs()
    cost = turnover * (costs.fee + costs.slippage)
    ret = held * bar_ret - cost
    if carry is not None:
        ret = ret + held * carry.reindex(close.index).fillna(0.0)
    ret = ret.iloc[1:]
    total, cagr, sharpe, mdd = _metrics(ret)
    split = int(len(ret) * (1 - oos_frac))
    oos_total, _, oos_sharpe, _ = _metrics(ret.iloc[split:]) if split < len(ret) else (np.nan, np.nan, np.nan, np.nan)
    return Result(
        name=name,
        equity=(1 + ret).cumprod(),
        position=position,
        total_return=total,
        cagr=cagr,
        sharpe=sharpe,
        max_dd=mdd,
        trades=int((turnover > 1e-9).sum()),
        exposure=float((held.abs() > 0).mean()),
        oos_return=oos_total,
        oos_sharpe=oos_sharpe,
    )


# --------------------------------------------------------------------------
# strategy families
# --------------------------------------------------------------------------

def buy_and_hold(close: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=close.index)


def sma_trend(close: pd.Series, fast: int = 20, slow: int = 100, allow_short: bool = False) -> pd.Series:
    f, s = close.rolling(fast).mean(), close.rolling(slow).mean()
    pos = (f > s).astype(float)
    if allow_short:
        pos = pos.where(f > s, -1.0)
    return pos.where(s.notna(), 0.0)


def breakout(close: pd.Series, lookback: int = 55, exit_lookback: int = 20) -> pd.Series:
    """Donchian: long on new N-day high, flat on M-day low."""
    hi = close.rolling(lookback).max().shift(1)
    lo = close.rolling(exit_lookback).min().shift(1)
    pos = pd.Series(np.nan, index=close.index)
    pos[close > hi] = 1.0
    pos[close < lo] = 0.0
    return pos.ffill().fillna(0.0)


def mean_reversion(close: pd.Series, window: int = 20, entry_z: float = -2.0, exit_z: float = 0.0) -> pd.Series:
    m, s = close.rolling(window).mean(), close.rolling(window).std()
    z = (close - m) / s
    pos = pd.Series(np.nan, index=close.index)
    pos[z < entry_z] = 1.0
    pos[z > exit_z] = 0.0
    return pos.ffill().fillna(0.0)


def vol_target(position: pd.Series, close: pd.Series, target_ann_vol: float = 0.30, window: int = 30) -> pd.Series:
    """Scale a 0/1 signal so realised vol is ~target. Keeps a small budget alive."""
    vol = close.pct_change().rolling(window).std() * np.sqrt(ANN)
    scale = (target_ann_vol / vol).clip(upper=1.0).fillna(0.0)
    return (position * scale).clip(-1, 1)


def funding_carry_daily(funding_8h: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Daily carry earned by a *short* perp (so sign is flipped for a long)."""
    if funding_8h is None or funding_8h.empty:
        return pd.Series(0.0, index=index)
    daily = funding_8h.resample("1D").sum()
    return daily.reindex(index).fillna(0.0)


def evaluate_instrument(
    close: pd.Series,
    category: str,
    funding_8h: pd.Series | None = None,
) -> list[Result]:
    costs = SPOT_COSTS if category == "spot" else PERP_COSTS
    is_perp = category in ("linear", "inverse")
    res = [
        run("buy_hold", close, buy_and_hold(close), costs),
        run("sma_20_100", close, sma_trend(close), costs),
        run("sma_20_100_voltgt", close, vol_target(sma_trend(close), close), costs),
        run("breakout_55_20", close, breakout(close), costs),
        run("meanrev_z20", close, mean_reversion(close), costs),
    ]
    if is_perp:
        res.append(run("sma_20_100_longshort", close, sma_trend(close, allow_short=True), costs))
        if funding_8h is not None and len(funding_8h):
            res.append(funding_carry(close, funding_8h, costs))
    return res


def funding_carry(close: pd.Series, funding_8h: pd.Series, perp_costs: Costs, oos_frac: float = 0.3) -> Result:
    """Delta-neutral carry: short perp + long spot, held the whole period.

    Price PnL cancels; the return is the funding a short collects (positive
    funding = longs pay shorts), minus the round trip on both legs.
    """
    close = close.dropna()
    carry = funding_carry_daily(funding_8h, close.index)
    start, end = funding_8h.index.min().floor("D"), funding_8h.index.max().floor("D")
    carry = carry[(carry.index >= start) & (carry.index <= end)]
    if carry.empty:
        carry = pd.Series(0.0, index=close.index[-1:])
    leg = perp_costs.fee + perp_costs.slippage + SPOT_COSTS.fee + SPOT_COSTS.slippage
    ret = carry.copy()
    ret.iloc[0] -= leg
    ret.iloc[-1] -= leg
    total, cagr, sharpe, mdd = _metrics(ret)
    split = int(len(ret) * (1 - oos_frac))
    oos_total, _, oos_sharpe, _ = _metrics(ret.iloc[split:]) if 0 < split < len(ret) else (np.nan, np.nan, np.nan, np.nan)
    return Result(
        name="funding_carry_neutral",
        equity=(1 + ret).cumprod(),
        position=pd.Series(1.0, index=ret.index),
        total_return=total,
        cagr=cagr,
        sharpe=sharpe,
        max_dd=mdd,
        trades=2,
        exposure=1.0,
        oos_return=oos_total,
        oos_sharpe=oos_sharpe,
        extra={"note": "price-neutral; funding income only, both legs' costs charged", "days": int(len(ret))},
    )


def evaluate_universe(
    history: dict[tuple[str, str], pd.DataFrame],
    funding: dict[str, pd.Series] | None = None,
    min_bars: int = 200,
) -> pd.DataFrame:
    rows = []
    for (cat, sym), df in history.items():
        if len(df) < min_bars:
            continue
        for r in evaluate_instrument(df.close, cat, (funding or {}).get(sym)):
            rows.append({"category": cat, "symbol": sym, **r.row()})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    bh = out[out.strategy == "buy_hold"].set_index(["category", "symbol"]).total_return
    out["vs_buy_hold"] = out.total_return.values - bh.reindex(list(zip(out.category, out.symbol))).values
    return out
