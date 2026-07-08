# Quant Bot Platform

An AI/ML-driven trading bot platform for crypto and forex: backtesting,
paper trading, and live execution, sharing one strategy and risk engine
across all three modes.

**Read this before the code:** if you came here expecting a system that
prints money, the honest answer up front is that this reference build's
own demo backtest *loses* money (-15.0% over 70 simulated days) once real
fees, slippage, and a genuine walk-forward validation are applied. That's
not a bug — it's the platform doing exactly what it's supposed to do:
tell you the truth about a strategy instead of a flattering one. See
[What the demo actually shows](#what-the-demo-actually-shows) below.

## Why this exists / how to read the results honestly

Social-media "AI trading bot" content — a certain 21-year-old's
$570K-in-61-days Claude Quant Bot screenshot, for instance — reliably
shares a few features: a live PnL counter that only goes up, "verified"
badges with no auditable methodology behind them, and zero mention of
transaction costs, slippage, or what happens on the drawdown. None of
that is a coincidence; the whole genre routes around the two things that
make a backtest honest:

1. **No lookahead.** A model tested on data it trained on will look
   incredible and mean nothing.
2. **Real costs.** A strategy that's profitable before fees and slippage
   is usually a strategy that's unprofitable after them.

This platform is built specifically so those two shortcuts aren't
available even by accident — see [Methodology](#methodology-the-parts-that-matter-more-than-the-model)
below. The tradeoff is that the numbers you get are real, which sometimes
means they're bad. This build's demo run is a legitimate example of that:
a modest, genuine signal (AUC 0.53-0.65 depending on symbol — better than
chance, nowhere near the video's implied near-certainty) that doesn't
clear real trading costs. That is useful information. A platform that
only ever shows you numbers you want to see isn't one you can trust with
real money.

## Architecture

The core design decision: **the same `Strategy` and `RiskManager` code
runs unmodified in backtest, paper, and live trading.** Only the data
source and execution destination change.

```
DataProvider  ──┐
                ├──► Strategy ──► RiskManager ──► ExecutionClient
Model/OOF ──────┘
```

| Mode      | Data source                          | Model used                              | Execution           |
|-----------|---------------------------------------|------------------------------------------|----------------------|
| Backtest  | Historical (synthetic/ccxt/OANDA)     | Walk-forward **out-of-fold** predictions | Simulated fills      |
| Paper     | Live feed (synthetic/ccxt/OANDA)      | Final model (never seen this data)       | Simulated fills      |
| Live      | Live feed (ccxt/OANDA)                | Final model (never seen this data)       | Real orders          |

The backtest deliberately does **not** call the live model — see
`backtest/oof_strategy.py` for why scoring the final model against data
it was trained on would be an in-sample fit dressed up as a backtest.

```
config/            YAML config (symbols, thresholds, risk limits)
data/providers/     DataProvider implementations: synthetic, ccxt (crypto), OANDA (forex)
features/           Hand-rolled technical indicators, no lookahead
models/             Dataset labeling, walk-forward training, inference
strategy/           Signal generation + risk management (sizing, kill switches)
backtest/           Event-driven backtest engine + performance metrics
execution/          Paper / live order execution clients
orchestrator/        The paper/live trading loop
api/                FastAPI backend serving the dashboard
dashboard/          index.html (live monitor) + report_template.html (backtest report)
scripts/            Entry points: generate data, train, backtest, paper trade, build report
tests/              pytest suite for the risk, dataset, metrics, and engine logic
```

## Quickstart

```bash
pip install -r requirements.txt --break-system-packages   # drop the flag if not needed on your system
cp .env.example .env                                        # fill in real keys later; safe to leave blank for now

python scripts/generate_sample_data.py   # synthetic data -- swap for real data later, see below
python scripts/train_model.py            # walk-forward training, saves models + OOF predictions
python scripts/run_backtest.py           # honest backtest using OOF predictions
python scripts/build_report.py           # builds results/report.html from the backtest above
python scripts/run_paper_trade.py        # short demo of the live-trading loop, using synthetic "live" data

uvicorn api.main:app --reload            # serves dashboard/index.html + results/report.html at http://localhost:8000
```

Run the test suite with `pytest tests/ -v`.

## What the demo actually shows

Running the pipeline above end-to-end (as shipped, no data or parameters
hand-picked afterward) on ~70 days of **synthetic** 5-minute OHLCV for
BTC/USDT, ETH/USDT, EUR/USD, and GBP/USD produced:

| Metric | Value |
|---|---|
| Total return | **-15.0%** |
| Sharpe (daily-resampled) | -6.49 |
| Sortino | -4.90 |
| Max drawdown | 15.0% (kill-switch limit: 15%) |
| Win rate | 29.3% |
| Profit factor | 0.41 |
| Trades | 796 over 70 days |

Per-symbol walk-forward AUC (out-of-fold, i.e. genuinely never seen
during training): BTC/USDT 0.542, ETH/USDT 0.532, EUR/USD 0.653, GBP/USD
0.616. Better than the 0.5 coin-flip baseline — the model found *some*
real, if modest, structure in the (synthetic) data — but not enough to
clear real transaction costs at the thresholds configured. **The max
drawdown kill-switch triggered partway through and halted trading for
the remainder of the backtest** — see the red status banner in
`results/report.html` — which is why the loss stops getting worse rather
than compounding for 70 days straight. That halt is the risk system
working as designed, not a failure separate from it.

These exact numbers will reproduce if you run the Quickstart commands
yourself (`data/providers/synthetic.py` is seeded and, as of this build,
genuinely deterministic across separate process runs — an earlier
version used Python's built-in `hash()` for a per-symbol seed offset,
which is randomized per-process for strings and made the "seed" silently
non-reproducible; it's now a fixed CRC32-based hash instead).

Open `results/report.html` for the full interactive breakdown, including
the walk-forward validation timeline for each symbol.

None of this means "the approach can't work" — it means this
combination of (synthetic data, this model, these thresholds, these
costs) didn't clear the bar, and the platform told you so instead of
hiding it. Tuning thresholds or trying other models is reasonable next
work; presenting a re-run with better-looking numbers as "the" result
without disclosing how many configurations were tried would reintroduce
exactly the kind of backtest-shopping this whole design tries to avoid.

## Going from synthetic to real data

Nothing about the pipeline changes — only the provider.

**Crypto**, via [ccxt](https://github.com/ccxt/ccxt) (100+ exchanges,
one interface):
```python
from data.providers.crypto_ccxt import CryptoCCXTProvider
provider = CryptoCCXTProvider(exchange_id="binance")  # or "coinbase", "kraken", "binanceus", ...
df = provider.historical("BTC/USDT", "5m", limit=5)     # no API key needed for public market data
```

**Forex**, via [OANDA's v20 API](https://developer.oanda.com/rest-live-v20/introduction/)
(free practice account, real historical + live data + execution under one login —
[get a token here](https://www.oanda.com/demo-account/tpa/personal_token)):
```python
from data.providers.forex_oanda import ForexOandaProvider
provider = ForexOandaProvider(api_token="YOUR_TOKEN", practice=True)
df = provider.historical("EUR_USD", "M5", limit=5)
```

**This has not been exercised against a live exchange or broker** — the
environment this was built in has no route to `api.binance.com` or
OANDA's servers (verified directly; see the code comments in
`data/providers/crypto_ccxt.py` and `forex_oanda.py`). Run the snippets
above on your own machine before trusting either connector.

Update `config/config.yaml`'s `data_provider` section and the constants
at the top of `scripts/*.py` accordingly once you've verified real-data
connectivity.

## Going live safely

1. **Backtest first** (above). Read the numbers, don't just check they're positive.
2. **Paper trade next**, for real — days or weeks, not minutes — against
   a real live data feed (swap `SyntheticProvider` for `CryptoCCXTProvider`/
   `ForexOandaProvider` in `scripts/run_paper_trade.py`, set realistic
   `poll_seconds`, and run it under a process supervisor). A strategy
   that looks fine in backtest can still fail in paper trading for
   reasons a backtest can't surface: data feed hiccups, a model that's
   confidently miscalibrated on data structurally different from its
   training window, a timezone assumption that was quietly wrong, etc.
3. **Only then, live** — and only with `LIVE_TRADING_CONFIRMED=yes`
   explicitly set in your environment. `execution/crypto_live.py` and
   `execution/forex_live.py` both refuse to place a single order without
   it. This is a deliberate friction point, not a bug to route around.

Risk controls that are on by default (`strategy/risk.py`), all
configurable in `config/config.yaml`:
- Volatility-scaled position sizing (risk a fixed % of equity per trade, sized off ATR)
- Hard cap on any single position and on total exposure across all positions
- Daily loss limit that blocks new entries for the rest of the day
- **Max drawdown kill-switch** that halts ALL trading and stays halted
  until a human calls `reset_halt()` — intentionally not automatic
- A no-trade rebalance band, so a continuously-varying confidence score
  doesn't generate a fee-paying trade on every single bar (see
  `RiskManager.is_significant_change` — an earlier version of this
  codebase didn't have this and churned so much it ran up ~30bps of cost
  on nearly every bar; the fix is in `strategy/risk.py` and used
  identically by both the backtester and the live orchestrator)

## Methodology (the parts that matter more than the model)

- **Cost-aware labels.** The model isn't trained to predict "up or down"
  — it's trained to predict "will this move exceed round-trip trading
  costs" (`models/dataset.py::make_labels`). A model that's 55% accurate
  at direction is worthless if the average winning move is smaller than
  what it costs to trade it.
- **Walk-forward validation with an embargo gap**, not random/k-fold
  cross-validation. Financial time series can't be shuffled — a random
  split leaks future information backward through autocorrelated bars.
  `models/dataset.py::walk_forward_folds` trains on an expanding window,
  skips an embargo gap equal to the label horizon, then tests on the
  next chunk, repeating forward through time.
- **Backtests replay out-of-fold predictions**, never the final
  production model, which is retrained on everything and would otherwise
  be scored on data it already saw (`backtest/oof_strategy.py`).
- **Realistic frictions**: configurable trading fees and slippage
  applied on every simulated fill, in both the backtest and paper engine.
- Not modeled, and worth knowing before you extend this: order-book
  depth / partial fills, funding rates on crypto perpetuals, margin
  interest, and network/exchange latency. A production system trading
  meaningful size would need all of these.

## Limitations

- This is a reference implementation, not a hardened production system.
  No auth on the API, no encrypted secrets storage beyond `.env`, no
  multi-region redundancy, no alerting/paging on failures.
- The synthetic data generator (`data/providers/synthetic.py`) is a
  regime-switching random walk for pipeline testing. It is not a
  calibrated model of any real market and shouldn't be treated as one.
- ~70 days of history (or even the couple of months you might have on
  hand before your own paper-trading track record is longer) is a thin
  sample for Sharpe/Sortino — daily-return-based risk metrics need
  months, ideally years, of live or paper history before they mean much
  statistically. `n_days_observed` is surfaced in every report precisely
  so a thin sample doesn't get mistaken for a robust one.
- Not financial advice. Nothing in this repo predicts, guarantees, or
  implies future returns of any kind, in any market.

## Docker

```bash
docker build -t quant-platform .
docker run -p 8000:8000 --env-file .env quant-platform
```

Runs the dashboard API only (`uvicorn api.main:app`). The trading loop
(`scripts/run_paper_trade.py` or a live equivalent) is intentionally a
separate process — see [Going live safely](#going-live-safely) for why
"the dashboard is running" and "the bot is trading" should never be the
same on/off switch.
