# Quant Bot Platform

An AI/ML-driven trading bot platform for crypto and forex: backtesting,
paper trading, and live execution, sharing one strategy and risk engine
across all three modes.

**Read this before the code:** if you came here expecting a system that
prints money, the honest answer up front is that this reference build's
own backtest *loses* money (-15.1% over 70 days of **real Binance data**)
once real fees, slippage, and a genuine walk-forward validation are
applied. That's not a bug — it's the platform doing exactly what it's
supposed to do: tell you the truth about a strategy instead of a
flattering one. See [What this build actually
shows](#what-this-build-actually-shows) below.

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
means they're bad. This build's own run is a legitimate example of that:
a modest, genuine signal (out-of-fold AUC 0.55-0.60 on real BTC/ETH/SOL/BNB
— better than chance, nowhere near the video's implied near-certainty)
that doesn't clear real trading costs. That is useful information. A
platform that only ever shows you numbers you want to see isn't one you
can trust with real money.

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
config/            YAML config (symbols, thresholds, risk limits) + loader.py resolving it
data/providers/     DataProvider implementations: synthetic, ccxt (crypto), OANDA (forex)
features/           Hand-rolled technical indicators, no lookahead
models/             Dataset labeling, walk-forward training, inference
strategy/           Signal generation + risk management (sizing, kill switches)
backtest/           Event-driven backtest engine + performance metrics
execution/          Paper / live order execution clients
orchestrator/        The paper/live trading loop
api/                FastAPI backend serving the dashboard
dashboard/          index.html (live monitor) + report_template.html (backtest report)
scripts/            Entry points: bootstrap (all-in-one), fetch, train, backtest, paper trade, build report
start.sh            Container entrypoint: bootstrap in background, then uvicorn
tests/              pytest suite for risk, dataset, metrics, engine, providers, API, bootstrap
```

## Quickstart

```bash
pip install -r requirements.txt --break-system-packages   # drop the flag if not needed on your system
cp .env.example .env                                        # fill in real keys later; safe to leave blank for now

python scripts/fetch_market_data.py      # REAL Binance 5m OHLCV -> data/_cache/ (no API key needed)
python scripts/train_model.py            # walk-forward training, saves models + OOF predictions
python scripts/run_backtest.py           # honest backtest using OOF predictions
python scripts/build_report.py           # builds results/report.html from the backtest above
python scripts/run_paper_trade.py        # short demo of the paper-trading loop against live prices

uvicorn api.main:app --reload            # serves dashboard/index.html + results/report.html at http://localhost:8000
```

`python scripts/bootstrap.py` runs those first four steps in order and skips
any whose output already exists — that's what a fresh deployment uses (see
[Deploying](#deploying)), and it's the convenient way to rebuild after a
config change. `python scripts/bootstrap.py --check` reports which artifacts
are present without building anything.

To run offline instead of against Binance, use
`python scripts/generate_sample_data.py` in place of the fetch step and set
`data_provider.crypto.type: synthetic` in `config/config.yaml`. The rest of
the pipeline is identical — that indirection is the point of
`data/providers/base.py`. The synthetic generator is seeded and genuinely
deterministic across separate process runs (an earlier version used Python's
built-in `hash()` for a per-symbol seed offset, which is randomized
per-process for strings and made the "seed" silently non-reproducible; it's
now a fixed CRC32-based hash instead).

Run the test suite with `pytest tests/ -v`.

## What this build actually shows

Running the pipeline above end-to-end, on **real Binance 5-minute OHLCV**
for BTC/USDT, ETH/USDT, SOL/USDT, and BNB/USDT — 20,000 bars per symbol,
2026-07-25 21:15 to 2026-10-03 07:50 UTC, aligned to a common grid — with
no data or parameters hand-picked afterward:

| Metric | Value |
|---|---|
| Total return | **-14.95%** |
| CAGR | -44.2% |
| Sharpe (daily-resampled) | -4.66 |
| Sortino | -4.41 |
| Max drawdown | 15.07% (kill-switch limit: 15%) |
| Win rate | 22.5% |
| Profit factor | 0.181 |
| Trades | 929 over 70 days |
| Peak gross exposure | 60.3% (cap: 60%) |

Per-symbol walk-forward AUC (out-of-fold, i.e. genuinely never seen
during training): BTC/USDT 0.600, ETH/USDT 0.574, BNB/USDT 0.563,
SOL/USDT 0.549. Better than the 0.5 coin-flip baseline — the model found
*some* real, if modest, structure in real market data — but not enough to
clear real transaction costs at the thresholds configured. **The max
drawdown kill-switch triggered partway through and halted trading for the
remainder of the backtest** — see the red status banner in
`results/report.html` — which is why the loss stops getting worse rather
than compounding for 70 days straight. That halt is the risk system
working as designed, not a failure separate from it.

This is the number that matters most, so it's worth stating plainly: the
model is *right more often than a coin flip* on real data, and the
strategy *still loses money*. That gap is transaction costs plus a
win rate that, at 22%, means the few winners don't pay for the many
losers. It is exactly the outcome the [Why this exists](#why-this-exists--how-to-read-results-honestly)
section warns about, and it's what a backtest is for.

### These numbers changed when the exposure cap was fixed

An earlier revision of this build computed total exposure once per bar and
then opened every symbol against that single stale figure, so four symbols
at 20% each cleared a 60% cap and left the book **80% exposed** — the cap was
decorative. Two defects were fixed under that heading, and the second moved
these numbers a second time, so they are worth keeping apart:

| | Cap unenforced | Cap enforced, symbols served in order | Cap enforced, book allocated proportionally |
|---|---|---|---|
| Total return | -15.12% | -15.13% | **-14.95%** |
| Sharpe | -6.29 | -5.16 | **-4.66** |
| Win rate | 25.3% | 19.7% | **22.5%** |
| Profit factor | 0.279 | 0.185 | 0.181 |
| Peak exposure | **80.1%** | 60.3% | 60.3% |
| Trades | — | 447 | 929 |

**The first defect was the cap itself.** Enforcing it properly made the
strategy look *worse*, and two things drove that — neither of which means
the signal "got worse":

1. **The cap decides which trades happen, and it decided arbitrarily.**
   Symbols were served in iteration order, so the book admitted the first
   three names and rejected the fourth — not because the fourth signal was
   worse, but because it came last. Measured by fixing the entry gate alone,
   this re-selection accounted for about two-thirds of the win-rate fall
   (25.3% → 21.8%).
2. **Trimming realizes losses that were previously left unrealized.** A book
   drifting over its cap gets sold back down, which converts paper drawdown
   into booked losses — the remaining third (21.8% → 19.7%). That is the
   risk system doing its job, and it costs money, as risk systems do.

**The second defect was that allocation rule.** Exposure is now rationed by
one factor applied to the whole book — `exposure_scale_to_cap` in
`strategy/risk.py` — so every symbol is scaled to a share of the cap rather
than the last one being rejected outright. Win rate recovers from 19.7% to
22.5%, most though not all of the 3.5pp that arbitrary re-selection cost,
and return and Sharpe come back with it.

The price is turnover, exactly as an earlier revision of this file predicted
it would be: trades go from 447 to 929. A book pinned at its cap is resized
whenever the mix of demand changes, where before it was left wherever the
first three symbols happened to put it. That extra cost is real, and it is
still smaller than what proportionality bought back.

Two subtleties fell out of implementing it, each of which cost a bug before
it was understood, and both are now pinned by tests. The no-trade band has
to be applied *after* the allocation rather than before it, because
allocation resizes positions by amounts the band never vetted — banding the
pre-allocation intent instead paid fees on 3,617 fills where ~600 were
warranted. And because the band lets a leg sit slightly *above* its share, a
book that reads as compliant can still be pushed over its cap by a new entry
opening at full allocated size: measured at **71.1%** against a 60% cap until
the banded book itself was checked against the cap, rather than the book the
bar started with.

The pre-fix numbers were *better-looking and less true*. A cap that reads
80% against a 60% limit isn't a conservative result, it's an uncontrolled
one, and the return it produced came from carrying a third more risk than
the configuration claimed. Anyone comparing these columns should prefer the
last.

The allocation is deliberately order-independent, which is the property the
test suite asserts directly: permuting the symbol list does not change the
book the portfolio ends up holding.

Open `results/report.html` for the full interactive breakdown, including
the walk-forward validation timeline for each symbol.

None of this means "the approach can't work" — it means this
combination of (real data, this model, these thresholds, these costs)
didn't clear the bar, and the platform told you so instead of hiding it.
Tuning thresholds or trying other models is reasonable next work;
presenting a re-run with better-looking numbers as "the" result without
disclosing how many configurations were tried would reintroduce exactly
the kind of backtest-shopping this whole design tries to avoid.

The earlier synthetic-data run of this same pipeline (BTC/ETH/EUR/USD/GBP/USD)
produced -15.0%, Sharpe -6.49, win rate 29.3% — a similar conclusion from a
different data source, which is itself mildly reassuring: the platform
isn't producing that result by accident of one dataset.

## Going from synthetic to real data

Nothing about the pipeline changes — only the provider. This is now the
**default** configuration: `scripts/fetch_market_data.py` pulls real Binance
OHLCV into `data/_cache/`, and `train_model.py` / `run_backtest.py` /
`build_report.py` read from that cache without knowing where it came from.

**Crypto**, via [ccxt](https://github.com/ccxt/ccxt) (100+ exchanges,
one interface):
```python
from data.providers.crypto_ccxt import CryptoCCXTProvider
provider = CryptoCCXTProvider(exchange_id="binance")  # or "coinbase", "kraken", "binanceus", ...
df = provider.historical("BTC/USDT", "5m", limit=5)     # no API key needed for public market data
```

This path is exercised: the results in [What this build actually
shows](#what-this-build-actually-shows) come from real Binance bars fetched
through it. One sharp edge worth knowing if you write your own fetch loop —
calling `fetch_ohlcv` with `since=None` returns the *most recent* bars, not
the oldest, so paging forward from there immediately runs off the end of the
series and silently returns a fraction of what you asked for. Always pass an
explicit `start`. `scripts/fetch_market_data.py` does, and
`tests/test_providers.py` pins the behaviour down.

**Forex**, via [OANDA's v20 API](https://developer.oanda.com/rest-live-v20/introduction/)
(free practice account, real historical + live data + execution under one login —
[get a token here](https://www.oanda.com/demo-account/tpa/personal_token)):
```python
from data.providers.forex_oanda import ForexOandaProvider
provider = ForexOandaProvider(api_token="YOUR_TOKEN", practice=True)
df = provider.historical("EUR_USD", "M5", limit=5)
```

**The forex path has not been exercised against live OANDA servers** — it
returns 403 without a token, and no token was available in the environment
this was built in. Forex therefore stays on the synthetic provider by
default. Verify the snippet above on your own machine before trusting the
connector.

Run forex as its **own pipeline pass**, not mixed with crypto. Every symbol
in a run must share a byte-identical timestamp index
(`backtest/engine.py` enforces this, and
`config/loader.py::align_to_common_index` satisfies it), and Binance stamps
5-minute bars at :00/:05 while the synthetic forex grid anchors to the current
wall clock. Pairing them in one equity curve produces a crash, not a result —
and blending real crypto with synthetic FX would make a misleading number
even if it ran.

Symbols and providers are resolved from `config/config.yaml` by
`config/loader.py`; there are no symbol lists or provider constants to edit
in `scripts/*.py`. To use real FX, set `data_provider.forex.type: oanda`,
fill `OANDA_API_TOKEN`, and run the pipeline with `--scope forex`.

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
  — enforced both when a position is opened and while it is held, since a
  position sized against the equity that existed at entry can represent a
  larger share of a smaller account later. When demand exceeds the cap the
  book is scaled by one factor applied to every position, so which symbols
  get held does not depend on the order they are iterated in
  (`strategy/risk.py`)  
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

The image runs `start.sh`, which launches `scripts/bootstrap.py` **in the
background** and then execs `uvicorn api.main:app`. Bootstrap fetches real
market data, trains, backtests, and builds the report; because it runs in the
background the port binds immediately, and `/api/health` reports
`bootstrap.state` (`running` with the current step, then `complete`) so the
dashboard can say "building artifacts" instead of showing an empty page. A
restart skips every step whose output already exists, so a redeploy costs a
few `stat()` calls rather than a full rebuild. Building artifacts into the
image instead would make it large and, worse, stale.

Only the dashboard API runs here. The trading loop
(`scripts/run_paper_trade.py` or a live equivalent) is intentionally a
separate process — see [Going live safely](#going-live-safely) for why
"the dashboard is running" and "the bot is trading" should never be the
same on/off switch. The one exception is the paper-trading loop, which the
dashboard can start and stop through the API; it is paper-only by
construction (see `api/paper_runner.py`) and cannot place a real order.

## Deploying

The platform works on any host that runs the Dockerfile. Three things to know:

1. **Artifacts are gitignored** (`data/_cache/`, `models/registry/`,
   `results/backtest_results.json`, `results/training_summary.json`,
   `runtime_state/`). This is correct for a repo — they're build output — but
   it means a fresh deploy starts with nothing. `scripts/bootstrap.py` exists
   to build them at startup. If your platform lets you set a start command,
   use `./start.sh` (or `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   plus a background `python scripts/bootstrap.py`). Running `uvicorn`
   directly still serves the dashboard, but it will show "no results yet"
   rather than data.
2. **Where you host decides which exchange you can reach.**
   `binance.com` returns HTTP 451 (Unavailable For Legal Reasons) to US IP
   addresses, so a host in the US — Render, Fly, most free tiers — cannot
   fetch from it at all. `data_provider.crypto.fallback_exchanges` in
   `config/config.yaml` lists venues tried in order when the primary fails;
   it ships as `[binanceus]`, the US-regulated venue serving the same four
   pairs. `scripts/fetch_market_data.py` prints which venue it used and
   records it in `data/_cache/_source.json`, which `run_backtest.py` copies
   into the results as `data_source.exchange` — so a run on Binance US is
   never silently reported as Binance.
3. **Tune the window for a free tier.** `history.bootstrap_bars` in
   `config/config.yaml` defaults to 6000 (~21 days of 5-minute bars), which
   keeps startup inside a typical free-tier health-check window. The 20,000-bar
   figures above take noticeably longer to build.

`/api/health` is the endpoint to watch: `ready: true` means a backtest and at
least one trained model are on disk. When something fails, `bootstrap.state`
is `failed` and `bootstrap.error` carries the last lines of the failing
step's output — deliberately, because on a hosted platform the API is often
the only surface you can reach without a dashboard login.
