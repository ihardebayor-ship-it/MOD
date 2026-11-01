"""Step 4: per-instrument statistics used to rank the universe.

Nothing here predicts the future. These numbers describe how an instrument
has behaved, so the backtester can be pointed at the ones where a given
strategy family has a chance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 365  # crypto trades every day


def _hurst(x: np.ndarray, max_lag: int = 20) -> float:
    """Rough Hurst exponent. >0.5 trending, <0.5 mean-reverting, ~0.5 random."""
    x = np.asarray(x, dtype=float)
    if len(x) < max_lag * 3:
        return float("nan")
    lags = range(2, max_lag)
    tau = [np.std(x[lag:] - x[:-lag]) for lag in lags]
    if any(t <= 0 for t in tau):
        return float("nan")
    slope = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
    return float(slope)


def instrument_features(df: pd.DataFrame, btc: pd.Series | None = None, funding: pd.Series | None = None) -> dict:
    close = df.close.dropna()
    if len(close) < 60:
        return {"bars": len(close)}
    r = np.log(close).diff().dropna()
    feats: dict = {
        "bars": len(close),
        "ann_vol": float(r.std() * np.sqrt(ANN)),
        "ret_30d": float(close.iloc[-1] / close.iloc[-min(30, len(close))] - 1),
        "ret_90d": float(close.iloc[-1] / close.iloc[-min(90, len(close))] - 1),
        "ret_total": float(close.iloc[-1] / close.iloc[0] - 1),
        "max_dd": float((close / close.cummax() - 1).min()),
        "hurst": _hurst(np.log(close.values)),
        "autocorr_1": float(r.autocorr(1)) if len(r) > 10 else float("nan"),
        "sma50_above": float(close.iloc[-1] > close.rolling(50).mean().iloc[-1]),
        "avg_turnover": float(df.turnover.tail(30).mean()) if "turnover" in df else float("nan"),
    }
    if btc is not None:
        both = pd.concat([r, np.log(btc).diff()], axis=1, join="inner").dropna()
        feats["corr_btc"] = float(both.iloc[:, 0].corr(both.iloc[:, 1])) if len(both) > 30 else float("nan")
    if funding is not None and len(funding):
        f = funding.tail(90)  # ~30 days at 8h
        feats["funding_mean_8h"] = float(f.mean())
        feats["funding_ann"] = float(f.mean() * 3 * ANN)
        feats["funding_pos_share"] = float((f > 0).mean())
    return feats


def build_feature_table(
    history: dict[tuple[str, str], pd.DataFrame],
    funding: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    btc = None
    for key in (("linear", "BTCUSDT"), ("spot", "BTCUSDT")):
        if key in history and not history[key].empty:
            btc = history[key].close
            break
    rows = []
    for (cat, sym), df in history.items():
        if df.empty:
            continue
        f = instrument_features(df, btc, (funding or {}).get(sym))
        rows.append({"category": cat, "symbol": sym, **f})
    if not rows:
        return pd.DataFrame(columns=["category", "symbol"]).set_index(["category", "symbol"])
    return pd.DataFrame(rows).set_index(["category", "symbol"]).sort_index()
