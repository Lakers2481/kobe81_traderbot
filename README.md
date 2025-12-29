# Kobe81 Traderbot — Backtesting, Paper, Live (Micro)

> **Version:** v2.3 — Comprehensive AI Briefing System
> **Last Updated:** 2025-12-29
> **Status:** Production Ready with LLM/ML/AI Integration

## Performance (v2.2 - Quant Interview Ready)

| Strategy | Win Rate | Profit Factor | Trades | Period |
|----------|----------|---------------|--------|--------|
| IBS+RSI | 59.9% | 1.46 | 867 | 2015-2024 |
| Turtle Soup | 61.0% | 1.37 | 305 | 2015-2024 |
| **Combined** | **60.2%** | **1.44** | **1,172** | 2015-2024 |

### Out-of-Sample Forward Test (2023-2024)
| Metric | Value |
|--------|-------|
| Win Rate | **64.1%** |
| Profit Factor | **1.60** |
| Trades | 192 |
| Net P&L | $1,696 |

*Forward test on unseen data shows BETTER performance than in-sample - strong evidence against overfitting.*

**Quick Replication:**
```bash
python scripts/backtest_dual_strategy.py --cap 200 --start 2015-01-01 --end 2024-12-31
```

**[Full Optimization Guide](docs/V2.2_OPTIMIZATION_GUIDE.md)** | **[Replication Steps](docs/STATUS.md#how-to-replicate-v22-results-critical---read-this)**

---

## CRITICAL: Strategy Verification (Read First)

**Use `backtest_dual_strategy.py` for ALL strategy verification.** This is the canonical test.

| Script | Use For | Strategy Class | Has Sweep Filter |
|--------|---------|----------------|------------------|
| `backtest_dual_strategy.py` | **Verification** | `DualStrategyScanner` | YES (0.3 ATR) |
| `run_wf_polygon.py` | Research only | `TurtleSoupStrategy` | NO |

**Why this matters:** Walk-forward uses a different code path that excludes the sweep strength filter, causing ~13% WR degradation. Always verify with `backtest_dual_strategy.py`.

**Verification Commands:**
```bash
# In-sample (should show ~60-61% WR)
python scripts/backtest_dual_strategy.py --start 2015-01-01 --end 2022-12-31 --cap 150

# Out-of-sample forward test (should show 60%+ WR)
python scripts/backtest_dual_strategy.py --start 2023-01-01 --end 2024-12-31 --cap 150
```

**See [docs/STATUS.md](docs/STATUS.md) for full investigation log with root cause analysis.**

---

Overview
- Strategies: Turtle Soup (liquidity sweep mean reversion) and IBS+RSI mean reversion (v2.2).
- Data: Polygon daily OHLCV with caching.
- Universe: optionable + liquid candidates filtered to final 900 with 10y coverage.
- Backtesting: deterministic next-bar fills; strategy-specific time stops (IBS 7 bars, Turtle 3 bars); no lookahead.
- Outputs: trade_list.csv, equity_curve.csv, summary.json per run/split.
- Walk-forward: rolling splits, side-by-side comparison (IBS_RSI vs TurtleSoup), HTML report.
- Execution: Alpaca IOC LIMIT submission (paper or live), kill switch, budgets, idempotency, audit log.

Project Map
- strategies/ - ICT Turtle Soup and IBS+RSI implementations
- backtest/ - engine + walk-forward
- data/ - providers (Polygon) + universe loader
- execution/ - Alpaca broker adapter (IOC limit)
- risk/ - PolicyGate (budgets, bounds)
- oms/ - order state + idempotency store
- core/ - hash-chain audit, config pin, structured logs
- monitor/ - health endpoints
- scripts/ - preflight, build/prefetch, WF/report, showdown, paper/live, reconcile, runner
- docs/ - COMPLETE_ROBOT_ARCHITECTURE.md, RUN_24x7.md, docs index

Requirements
- Python 3.11+
- Install: `pip install -r requirements.txt`
- Env: set in `./.env`
  - `POLYGON_API_KEY=...`
  - `ALPACA_API_KEY_ID=...`
  - `ALPACA_API_SECRET_KEY=...`
  - `ALPACA_BASE_URL=https://paper-api.alpaca.markets` (paper) or live endpoint

Quick Start
1) Preflight (keys + config pin + broker probe)
   python scripts/preflight.py --dotenv ./.env

2) Build the 900-stock universe with proofs
   python scripts/build_universe_polygon.py --candidates data/universe/optionable_liquid_candidates.csv --start 2015-01-01 --end 2024-12-31 --min-years 10 --cap 900 --concurrency 3 --cache data/cache --out data/universe/optionable_liquid_900.csv --dotenv ./.env

3) Prefetch EOD bars (faster WF)
   python scripts/prefetch_polygon_universe.py --universe data/universe/optionable_liquid_900.csv --start 2015-01-01 --end 2024-12-31 --cache data/cache --concurrency 3 --dotenv ./.env

4) Walk-forward comparison and report
   python scripts/run_wf_polygon.py --universe data/universe/optionable_liquid_900.csv --start 2015-01-01 --end 2024-12-31 --train-days 252 --test-days 63 --cap 900 --outdir wf_outputs --cache data/cache --dotenv ./.env
   python scripts/aggregate_wf_report.py --wfdir wf_outputs --out wf_outputs/wf_report.html

5) Showdown (full-period side-by-side)
   python scripts/run_showdown_polygon.py --universe data/universe/optionable_liquid_900.csv --start 2015-01-01 --end 2024-12-31 --cap 900 --outdir showdown_outputs --cache data/cache --dotenv ./.env

Paper and Live Trading (IOC LIMIT)
- Paper (micro budgets):
  python scripts/run_paper_trade.py --universe data/universe/optionable_liquid_900.csv --start 2015-01-01 --end 2024-12-31 --cap 50 --dotenv ./.env

- Live (micro budgets; set ALPACA_BASE_URL to live in .env):
  python scripts/run_live_trade_micro.py --universe data/universe/optionable_liquid_900.csv --start 2015-01-01 --end 2024-12-31 --cap 10 --dotenv ./.env

Evidence Artifacts
- wf_outputs/wf_summary_compare.csv â€” strategy side-by-side KPIs
- wf_outputs/<strategy>/split_NN/{trade_list.csv,equity_curve.csv,summary.json}
- showdown_outputs/showdown_summary.csv, showdown_report.html
- data/universe/optionable_liquid_900.csv and `.full.csv` (coverage, ADV, options proofs)
- logs/events.jsonl (structured), state/hash_chain.jsonl (audit)

24/7 Runner
- Paper example:
  python scripts/runner.py --mode paper --universe data/universe/optionable_liquid_900.csv --cap 50 --scan-times 09:35,10:30,15:55 --lookback-days 540 --dotenv ./.env
- Live example:
  python scripts/runner.py --mode live --universe data/universe/optionable_liquid_900.csv --cap 10 --scan-times 09:35,10:30,15:55 --lookback-days 540 --dotenv ./.env
 - Task Scheduler setup: see docs/RUN_24x7.md

Time Zones
- Policy: All schedules operate in ET; all displays (UI and Telegram) show CT and ET in 12-hour format (e.g., "8:45 AM CT | 9:45 AM ET").
- Trading operations (schedules, scan dates) run on New York time (ET). Displays (live dashboard) default to Central Time (CT) with AM/PM.
- Windows Task registration (ops/windows/register_all_tasks.ps1) registers jobs at ET hours converted to your local clock (handles DST automatically).

Safety
- Kill switch: create file `state/KILL_SWITCH` to halt submissions.
- Policy Gate: per-order and daily budgets ($75 / $1,000), price bounds, shorts disabled by default.
- Audit: verify `python scripts/verify_hash_chain.py`.
- Reconciliation: `python scripts/reconcile_alpaca.py --dotenv ./.env`.

Config-Gated Features (config/base.yaml)
All features below are disabled by default. Enable in `config/base.yaml`:

1) Commissions/Fees Model (backtest only)
   - Set `backtest.commissions.enabled: true`
   - Configure per-share, BPS, SEC/TAF fees
   - Outputs gross_pnl, net_pnl, total_fees in summary.json

2) LULD/Volatility Clamp (execution)
   - Set `execution.clamp.enabled: true`
   - Fixed percentage or ATR-based clamping
   - Prevents limit prices from exceeding LULD bands

3) Order Rate Limiter + Retry
   - Set `execution.rate_limiter.enabled: true`
   - Token bucket with 120 orders/min capacity
   - Exponential backoff retry on 429 errors

4) Earnings Proximity Filter
   - Set `filters.earnings.enabled: true`
   - Skips signals 2 days before / 1 day after earnings
   - Caches earnings dates in state/earnings_cache.json

5) Metrics Endpoint
   - Set `health.metrics.enabled: true` (default: true)
   - GET /metrics returns request counters and performance stats
   - Includes uptime, WR, PF, Sharpe from last run

6) Regime Filter (SPY-based)
   - Set `regime_filter.enabled: true`
   - Trend gate: SPY close > SMA(200), fast SMA(20) > slow SMA(200)
   - Volatility gate: realized vol <= max_ann_vol threshold
   - Module: `core/regime_filter.py`

7) Signal Selection (Top-N Ranking)
   - Disabled in this two-strategy setup; Top-3 logic is handled inside `scripts/scan.py` (2×ICT + 1×IBS_RSI)

8) Volatility-Targeted Sizing
   - Set `sizing.enabled: true`
   - Formula: qty = (risk_pct × equity) / (entry - stop)
   - Default risk_per_trade_pct: 0.5% (0.005)
   - Requires stop_loss in signal for calculation

Robustness Tools
- Parameter Optimization:
  python scripts/optimize.py --universe data/universe/optionable_liquid_900.csv --start 2015-01-01 --end 2024-12-31 --cap 100 --outdir optimize_outputs --dotenv ./.env

- Monte Carlo Robustness Testing:
  python scripts/monte_carlo.py --trades wf_outputs/ibs_rsi/split_00/trade_list.csv --iterations 1000 --outdir monte_carlo_outputs

Crypto (Backtest-Only)
Research-only crypto backtesting using Polygon hourly bars. No live execution.

Universe Files:
- data/universe/crypto_top3.csv (BTC, ETH, SOL)
- data/universe/crypto_top10.csv (top 10 by market cap)

walk-forward:
  python scripts/run_wf_crypto.py --universe data/universe/crypto_top3.csv --start 2020-01-01 --end 2024-12-31 --train-days 252 --test-days 63 --outdir wf_outputs_crypto --cache data/cache/crypto --dotenv ./.env

Report:
  python scripts/aggregate_wf_report.py --wfdir wf_outputs_crypto --out wf_outputs_crypto/wf_report.html

Showdown:
  python scripts/run_showdown_crypto.py --universe data/universe/crypto_top10.csv --start 2020-01-01 --end 2024-12-31 --outdir showdown_outputs_crypto --cache data/cache/crypto --dotenv ./.env

---

## AI Briefing System (v2.3 - NEW)

Comprehensive 3-phase briefing system with Claude LLM integration for intelligent market analysis:

| Phase | Time (ET) | Purpose | Output |
|-------|-----------|---------|--------|
| **PRE_GAME** | 08:00 | Morning game plan | Regime analysis, Top-3 picks, TOTD, action steps |
| **HALF_TIME** | 12:00 | Midday check | Position P&L, adjustments, afternoon plan |
| **POST_GAME** | 16:00 | EOD reflection | Performance review, lessons, next-day setup |

**Generate Briefings:**
```bash
# Morning briefing (pre-market)
python scripts/generate_briefing.py --phase pregame --dotenv ./.env

# Midday briefing
python scripts/generate_briefing.py --phase halftime --dotenv ./.env

# End-of-day briefing
python scripts/generate_briefing.py --phase postgame --dotenv ./.env

# With Telegram notification
python scripts/generate_briefing.py --phase pregame --telegram --dotenv ./.env
```

**Briefing Features:**
- **Regime Analysis**: HMM-based market regime detection with probability breakdown
- **Market Mood**: VIX + sentiment fusion for emotional state assessment
- **News Integration**: Real-time news with LLM interpretation
- **Trade Narratives**: Claude-generated analysis for each signal
- **Position Tracking**: Live P&L and hold/adjust/exit recommendations
- **Learning System**: Hypothesis generation and lesson extraction

**Output Files:**
- `reports/pregame_YYYYMMDD.json` + `.md`
- `reports/halftime_YYYYMMDD.json` + `.md`
- `reports/postgame_YYYYMMDD.json` + `.md`

**Key Modules:**
- `cognitive/game_briefings.py` - Main briefing engine
- `cognitive/llm_trade_analyzer.py` - Signal narrative generation
- `altdata/news_processor.py` - News fetching and sentiment
- `altdata/market_mood_analyzer.py` - VIX + sentiment fusion

---

Interview Quick Start (3 commands)
- Ensure `./.env` exists with your keys (see Requirements above).
- Run the quick test (50 stocks, ~12 months):
  python scripts/interview_quick_test.py --dotenv ./.env
- Open the HTML report: `wf_outputs_interview_quick/wf_report.html`
- Share the summary JSON: `INTERVIEW_SUMMARY.json`







Note: For canonical commands and configuration, use README.md, AI_HANDOFF_PROMPT.md, and scripts under scripts/. Some deep reference docs or .claude/ content may include legacy examples; when in doubt, follow README and AI_HANDOFF_PROMPT as source of truth.



