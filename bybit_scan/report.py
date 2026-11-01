"""Step 6: turn the tables into a readable markdown report."""

from __future__ import annotations

import pandas as pd


def _pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:+.1f}%"


def _num(x: float, d: int = 2) -> str:
    return "n/a" if pd.isna(x) else f"{x:.{d}f}"


def _table(df: pd.DataFrame, cols: list[str], fmt: dict | None = None, n: int = 25) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.head(n).iterrows():
        cells = []
        for c in cols:
            v = r.get(c, None)
            f = fmt.get(c)
            if v is None or (f and not isinstance(v, (int, float))):
                cells.append("")
            else:
                cells.append(f(v) if f else str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def build_report(
    inv_summary: pd.DataFrame,
    universe: pd.DataFrame,
    feats: pd.DataFrame,
    bt: pd.DataFrame,
    budget: float | None = None,
) -> str:
    parts = ["# Bybit full-market scan\n"]

    parts.append("## 1. What Bybit exposes through the API\n")
    parts.append(_table(inv_summary.reset_index(), ["category", "instruments", "trading", "turnover24h_usd", "median_spread_bps"],
                        {"turnover24h_usd": lambda v: f"${v/1e6:,.0f}M", "median_spread_bps": lambda v: _num(v, 1)}))
    parts.append("\nOptions are inventoried but not backtested: with a small budget the bid/ask on most strikes eats any edge.\n")

    parts.append(f"\n## 2. Tradeable universe ({len(universe)} instruments after liquidity filters)\n")
    cat_counts = universe.category.value_counts().to_dict()
    parts.append(", ".join(f"{k}: {v}" for k, v in cat_counts.items()) + "\n\n")
    parts.append(_table(universe, ["category", "symbol", "turnover24h", "spread_bps", "openInterestValue", "fundingRate"],
                        {"turnover24h": lambda v: f"${v/1e6:,.1f}M", "spread_bps": lambda v: _num(v, 1),
                         "openInterestValue": lambda v: "" if pd.isna(v) else f"${v/1e6:,.0f}M",
                         "fundingRate": lambda v: "" if pd.isna(v) else f"{v*100:.4f}%"}, n=30))

    if not feats.empty:
        f = feats.reset_index()
        parts.append("\n## 3. Behaviour (last ~2 years, daily bars)\n")
        parts.append("Hurst > 0.55 leans trending, < 0.45 leans mean-reverting. `funding_ann` is what a short perp would earn per year at the last 30 days' funding, before costs.\n\n")
        parts.append("**Strongest trenders**\n\n" + _table(f.sort_values("hurst", ascending=False), ["category", "symbol", "hurst", "ann_vol", "ret_90d", "max_dd", "corr_btc"],
                        {"hurst": _num, "ann_vol": _pct, "ret_90d": _pct, "max_dd": _pct, "corr_btc": _num}, n=15))
        parts.append("\n**Most mean-reverting**\n\n" + _table(f.sort_values("hurst"), ["category", "symbol", "hurst", "autocorr_1", "ann_vol", "max_dd"],
                        {"hurst": _num, "autocorr_1": _num, "ann_vol": _pct, "max_dd": _pct}, n=15))
        if "funding_ann" in f:
            parts.append("\n**Richest funding (carry candidates)**\n\n" + _table(f.dropna(subset=["funding_ann"]).sort_values("funding_ann", ascending=False),
                        ["symbol", "funding_ann", "funding_pos_share", "ann_vol"],
                        {"funding_ann": _pct, "funding_pos_share": _pct, "ann_vol": _pct}, n=15))

    if not bt.empty:
        parts.append("\n## 4. Backtests (fees + slippage included, last 30% of history is out-of-sample)\n")
        parts.append("Sort key is **out-of-sample Sharpe**, not total return. A strategy that only works in-sample is curve-fit.\n\n")
        strat = bt.groupby("strategy").agg(
            n=("symbol", "size"),
            median_ret=("total_return", "median"),
            median_sharpe=("sharpe", "median"),
            median_oos_sharpe=("oos_sharpe", "median"),
            share_beats_bh=("vs_buy_hold", lambda s: (s > 0).mean()),
            median_dd=("max_dd", "median"),
        ).sort_values("median_oos_sharpe", ascending=False).reset_index()
        parts.append("**By strategy family (median across instruments)**\n\n" + _table(strat,
                        ["strategy", "n", "median_ret", "median_sharpe", "median_oos_sharpe", "share_beats_bh", "median_dd"],
                        {"median_ret": _pct, "median_sharpe": _num, "median_oos_sharpe": _num, "share_beats_bh": _pct, "median_dd": _pct}))
        best = bt[bt.strategy != "buy_hold"].sort_values("oos_sharpe", ascending=False)
        parts.append("\n**Best instrument/strategy pairs out-of-sample**\n\n" + _table(best,
                        ["category", "symbol", "strategy", "total_return", "oos_return", "sharpe", "oos_sharpe", "max_dd", "trades", "vs_buy_hold"],
                        {"total_return": _pct, "oos_return": _pct, "sharpe": _num, "oos_sharpe": _num, "max_dd": _pct, "vs_buy_hold": _pct}, n=30))

    parts.append("\n## 5. Read this before trading any of it\n")
    parts.append(
        "- Past performance here is descriptive, not predictive. Two years of crypto history is one regime.\n"
        "- Anything with fewer than ~30 trades has a Sharpe that is mostly noise.\n"
        "- Funding carry looks safe but needs both legs filled and funding can flip negative for weeks.\n"
        "- Leverage multiplies drawdown faster than it multiplies return. A 50% drawdown needs +100% to recover.\n"
    )
    if budget:
        parts.append(f"- With a ${budget:,.0f} budget, instruments with `minOrderAmt` above ~1% of budget cannot be position-sized sensibly.\n")
    return "".join(parts)
