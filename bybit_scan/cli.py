"""Command line: python -m bybit_scan scan --out out/ [--budget 1000]"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

from .backtest import evaluate_universe
from .client import MAINNET, TESTNET, BybitError, BybitPublic
from .data import fetch_all_funding, fetch_history
from .features import build_feature_table
from .inventory import fetch_inventory, summarize_inventory
from .report import build_report
from .universe import UniverseConfig, filter_universe


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bybit_scan", description="Scan every Bybit instrument, filter, rank, backtest.")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="out", help="output directory")
    common.add_argument("--testnet", action="store_true")
    common.add_argument("--cache-dir", default=".cache/bybit")

    sub.add_parser("ping", parents=[common], help="check that Bybit is reachable from this machine")
    sub.add_parser("inventory", parents=[common], help="list every instrument across all categories")

    s = sub.add_parser("scan", parents=[common], help="inventory → filter → history → features → backtests → report")
    s.add_argument("--budget", type=float, default=None, help="your fixed budget in USD (only affects report notes)")
    s.add_argument("--days", type=int, default=730)
    s.add_argument("--min-turnover", type=float, default=2_000_000)
    s.add_argument("--max-spread-bps", type=float, default=15.0)
    s.add_argument("--max-symbols", type=int, default=None, help="cap universe size (debug)")
    s.add_argument("--workers", type=int, default=6)

    a = p.parse_args(argv)
    client = BybitPublic(TESTNET if a.testnet else MAINNET, cache_dir=a.cache_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        if a.cmd == "ping":
            print("server time:", client.server_time(), "— reachable")
            return 0

        print("inventory: fetching every instrument…")
        inv = fetch_inventory(client)
        inv.to_csv(out / "inventory.csv", index=False)
        summ = summarize_inventory(inv)
        print(summ.to_string())
        if a.cmd == "inventory":
            return 0

        cfg = UniverseConfig(min_turnover_usd=a.min_turnover, max_spread_bps=a.max_spread_bps)
        uni = filter_universe(inv, cfg)
        if a.max_symbols:
            uni = uni.head(a.max_symbols)
        uni.to_csv(out / "universe.csv", index=False)
        print(f"universe: {len(uni)} tradeable instruments ({uni.category.value_counts().to_dict()})")

        print("history: downloading daily candles…")
        hist = fetch_history(client, uni, "D", a.days, a.workers)
        print("funding: downloading funding history for perps…")
        fund = fetch_all_funding(client, uni, min(a.days, 365), a.workers)
        with open(out / "history.pkl", "wb") as fh:
            pickle.dump({"history": hist, "funding": fund}, fh)

        feats = build_feature_table(hist, fund)
        feats.to_csv(out / "features.csv")
        print("backtests…")
        bt = evaluate_universe(hist, fund)
        bt.to_csv(out / "backtests.csv", index=False)

        md = build_report(summ, uni, feats, bt, a.budget)
        (out / "report.md").write_text(md)
        print(f"\nreport written to {out / 'report.md'}")
        return 0
    except BybitError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
