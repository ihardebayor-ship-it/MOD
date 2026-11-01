# bybit-scan

Scan **every** instrument Bybit exposes through its V5 API — spot (including
tokenized stocks / RWA pairs), USDT & USDC perpetuals, dated futures, inverse
contracts, pre-market contracts and options — then filter to what a small
budget can actually trade, describe how each one behaves, and backtest a few
simple strategy families with real fees and slippage.

No API key is required. Everything uses public endpoints.

## What is and isn't reachable

| Bybit product | In scope | How |
|---|---|---|
| Spot, tokenized stocks, RWA | ✅ | `category=spot` |
| USDT/USDC perps & futures, gold/FX perps | ✅ | `category=linear` |
| Inverse perps/futures | ✅ | `category=inverse` |
| Pre-market futures | ✅ (listed, not traded) | `status=PreLaunch` |
| Options | ✅ (listed, not backtested) | `category=option&baseCoin=All` |
| TradFi MT5 (forex/CFDs) | ❌ | separate MT5 platform, no V5 API |
| Earn / Launchpool / Copy trading / P2P | ❌ | not market instruments |

**Bybit geo-blocks the US and some other regions.** If `ping` prints an HTTP 403,
run this from a permitted country or a VPS in one (Singapore, Frankfurt, …).

## Quick start

```bash
pip install -r requirements.txt
python -m bybit_scan ping                       # can this machine reach Bybit?
python -m bybit_scan inventory --out out        # every instrument → out/inventory.csv
python -m bybit_scan scan --out out --budget 1000
```

`scan` takes roughly 5–15 minutes the first time (a few thousand requests, rate
limited to be polite). Responses are cached under `.cache/bybit`, so re-runs are fast.

Outputs in `out/`:

| file | contents |
|---|---|
| `inventory.csv` | every instrument + 24h ticker (turnover, spread, OI, funding) |
| `universe.csv` | the liquid subset that survives the filters |
| `features.csv` | volatility, trend/mean-reversion stats, BTC correlation, funding carry |
| `backtests.csv` | every instrument × strategy with in- and out-of-sample metrics |
| `report.md` | the human-readable summary — read this one |

## Strategy families tested

- `buy_hold` – the baseline everything must beat
- `sma_20_100` – long when the 20-day average is above the 100-day
- `sma_20_100_voltgt` – same, sized so realised volatility ≈ 30%/yr
- `sma_20_100_longshort` – perps only, short when below
- `breakout_55_20` – Donchian: buy 55-day highs, exit 20-day lows
- `meanrev_z20` – buy when price is 2σ below its 20-day mean, exit at the mean
- `funding_carry_neutral` – perps only: short perp + long spot, collect funding

Costs: spot taker 0.10%, perp taker 0.055%, plus 0.05% slippage per side.
The last 30% of history is held out as out-of-sample; **rank by `oos_sharpe`**.

## Tuning the filters

```bash
python -m bybit_scan scan --min-turnover 5000000 --max-spread-bps 8 --days 365
```

Defaults: ≥ $2M 24h turnover, ≤ 15 bps spread, ≥ $1M open interest for perps,
USDT/USDC quotes only, no "innovation zone" tokens, no dated futures.

## Tests

```bash
pip install pytest
pytest
```

The tests run fully offline against a fake Bybit server with synthetic markets.

## Honest notes

- Nothing in here predicts anything. It describes what *has* happened after costs.
- A fixed budget is multiplied by compounding returns **and** by compounding
  drawdowns. Position sizing and a hard stop matter more than instrument choice.
- Never give an API key withdrawal permission. Use a sub-account that holds
  only the budget you are prepared to lose.
