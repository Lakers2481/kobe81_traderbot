# Kobe81 Trading System - Full Architecture

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    KOBE81 TRADING SYSTEM - FULL ARCHITECTURE                               ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────── DATA LAYER ───────────────────────────────────────────┐
│                                                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                    │
│  │  Polygon.io  │    │    Stooq     │    │ Yahoo Fin    │    │   Binance    │                    │
│  │   (EOD API)  │    │  (Free EOD)  │    │  (Free EOD)  │    │   (Crypto)   │                    │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                    │
│         │                   │                   │                   │                            │
│         └───────────────────┴───────────────────┴───────────────────┘                            │
│                                         │                                                        │
│                                         ▼                                                        │
│                            ┌────────────────────────┐                                            │
│                            │   data/providers/      │                                            │
│                            │   polygon_eod.py       │◄──── CSV Cache (data/cache/)               │
│                            │   (OHLCV + Volume)     │                                            │
│                            └───────────┬────────────┘                                            │
│                                        │                                                         │
│         ┌──────────────────────────────┼──────────────────────────────┐                          │
│         ▼                              ▼                              ▼                          │
│  ┌─────────────┐             ┌─────────────────┐            ┌─────────────────┐                  │
│  │  Universe   │             │  Data Quality   │            │  Alt Data       │                  │
│  │  900 stocks │             │  preflight/     │            │  altdata/       │                  │
│  │  10yr hist  │             │  data_quality   │            │  sentiment.py   │                  │
│  └─────────────┘             └─────────────────┘            └─────────────────┘                  │
│                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────── RESEARCH LAYER ───────────────────────────────────────────┐
│                                                                                                   │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐    ┌────────────────┐            │
│  │ research/      │    │ research/      │    │ research/      │    │ data_explor/   │            │
│  │ features.py    │    │ alphas.py      │    │ screener.py    │    │ feature_imp.py │            │
│  │ (25 features)  │    │ (18 alphas)    │    │ (WF screening) │    │ (importance)   │            │
│  └───────┬────────┘    └───────┬────────┘    └───────┬────────┘    └───────┬────────┘            │
│          │                     │                     │                     │                     │
│          └─────────────────────┴─────────────────────┴─────────────────────┘                     │
│                                              │                                                    │
│                                              ▼                                                    │
│                               ┌──────────────────────────┐                                        │
│                               │    Feature Discovery     │                                        │
│                               │    data_exploration/     │                                        │
│                               │    + Data Registry       │                                        │
│                               └──────────────────────────┘                                        │
│                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────── STRATEGY LAYER ───────────────────────────────────────────┐
│                                                                                                   │
│     ┌─────────────────────────────┐              ┌─────────────────────────────┐                 │
│     │   DONCHIAN BREAKOUT         │              │   ICT TURTLE SOUP           │                 │
│     │   strategies/donchian/      │              │   strategies/ict/           │                 │
│     ├─────────────────────────────┤              ├─────────────────────────────┤                 │
│     │ • 20-day channel breakout   │              │ • Liquidity sweep reversal  │                 │
│     │ • SMA(200) trend filter     │              │ • SMA(200) trend filter     │                 │
│     │ • ATR(14) Wilder smoothing  │              │ • ATR(14) Wilder smoothing  │                 │
│     │ • 2x ATR stop + 5-bar exit  │              │ • 2x ATR stop + 5-bar exit  │                 │
│     └─────────────┬───────────────┘              └─────────────┬───────────────┘                 │
│                   │                                            │                                 │
│                   └──────────────────┬─────────────────────────┘                                 │
│                                      ▼                                                           │
│                         ┌─────────────────────────┐                                              │
│                         │  generate_signals(df)   │◄──── Shifted indicators (no lookahead)      │
│                         │  → timestamp, symbol,   │                                              │
│                         │    side, entry, stop,   │                                              │
│                         │    take_profit, reason  │                                              │
│                         └───────────┬─────────────┘                                              │
│                                     │                                                            │
└─────────────────────────────────────┼────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────── ML / META-MODEL ──────────────────────────────────────────┐
│                                                                                                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │ ml/meta_model   │    │ ml/confidence   │    │ monitor/        │    │ monitor/        │        │
│  │ .py             │───►│ _gate.py        │───►│ calibration.py  │───►│ drift_detector  │        │
│  │ (ensemble)      │    │ (0.6 threshold) │    │ (Brier score)   │    │ (perf tracking) │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│                                     │                                                            │
│                                     ▼                                                            │
│                    ┌────────────────────────────────┐                                            │
│                    │     TRADE OF THE DAY (TOTD)    │                                            │
│                    │     Confidence-gated pick      │                                            │
│                    │     + Sentiment blending       │                                            │
│                    └────────────────────────────────┘                                            │
│                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────── BACKTEST ENGINE ──────────────────────────────────────────┐
│                                                                                                   │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐    ┌────────────────┐            │
│  │ backtest/      │    │ backtest/      │    │ experiments/   │    │ options/       │            │
│  │ engine.py      │    │ walk_forward   │    │ registry.py    │    │ backtest.py    │            │
│  │ (FIFO P&L)     │    │ .py (WF splits)│    │ (reproducible) │    │ (synth B-S)    │            │
│  └───────┬────────┘    └───────┬────────┘    └───────┬────────┘    └───────┬────────┘            │
│          │                     │                     │                     │                     │
│          │    ┌────────────────┴─────────────────────┴─────────────────────┘                     │
│          │    │                                                                                  │
│          ▼    ▼                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐                        │
│  │                        WALK-FORWARD VALIDATION                        │                        │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │                        │
│  │  │Train 252│►│Test 63  │►│Train 252│►│Test 63  │►│Train 252│► ...    │                        │
│  │  │  days   │ │  days   │ │  days   │ │  days   │ │  days   │         │                        │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘         │                        │
│  │                                                                       │                        │
│  │  Output: wf_outputs/{strategy}/split_NN/{trade_list, equity_curve}   │                        │
│  └──────────────────────────────────────────────────────────────────────┘                        │
│                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────── RISK MANAGEMENT ──────────────────────────────────────────┐
│                                                                                                   │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐   │
│  │    PolicyGate     │   │   LiquidityGate   │   │   Circuit Breaker │   │   Kill Switch     │   │
│  │  risk/policy_gate │   │  risk/liquidity   │   │  selfmonitor/     │   │  state/KILL_SWITCH│   │
│  ├───────────────────┤   ├───────────────────┤   ├───────────────────┤   ├───────────────────┤   │
│  │ • $75/order max   │   │ • $100k min ADV   │   │ • Max daily loss  │   │ • Emergency halt  │   │
│  │ • $1k/day budget  │   │ • 0.5% max spread │   │ • Consec losses   │   │ • Blocks all      │   │
│  │ • Auto-reset      │   │ • Impact limits   │   │ • Error threshold │   │   order placement │   │
│  └─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘   │
│            │                       │                       │                       │             │
│            └───────────────────────┴───────────────────────┴───────────────────────┘             │
│                                              │                                                    │
│                                              ▼                                                    │
│                              ┌───────────────────────────────┐                                    │
│                              │      ALL GATES MUST PASS      │                                    │
│                              │      before order placement   │                                    │
│                              └───────────────────────────────┘                                    │
│                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────── COMPLIANCE ENGINE ────────────────────────────────────────┐
│                                                                                                   │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐                           │
│  │  Rules Engine   │      │ Prohibited List │      │   Audit Trail   │                           │
│  │ compliance/     │      │ compliance/     │      │ compliance/     │                           │
│  │ rules_engine.py │      │ prohibited_list │      │ audit_trail.py  │                           │
│  ├─────────────────┤      ├─────────────────┤      ├─────────────────┤                           │
│  │ • Max pos size  │      │ • Earnings ban  │      │ • Hash-verified │                           │
│  │ • PDT rule      │      │ • News events   │      │ • Every action  │                           │
│  │ • No penny stks │      │ • Volatility    │      │ • Tamper-proof  │                           │
│  │ • Trading hours │      │ • Auto-expire   │      │ • JSONL format  │                           │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘                           │
│                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────── EXECUTION LAYER ──────────────────────────────────────────┐
│                                                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              ORDER MANAGEMENT SYSTEM (OMS)                                 │   │
│  │                                                                                            │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                        │   │
│  │  │  OrderRecord    │    │ IdempotencyStore│    │  Order State    │                        │   │
│  │  │  oms/order_     │    │ oms/idempotency │    │  Tracking       │                        │   │
│  │  │  state.py       │    │ _store.py       │    │  (no dupes)     │                        │   │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘                        │   │
│  │                                                                                            │   │
│  └───────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                                    │
│                                              ▼                                                    │
│                              ┌───────────────────────────────┐                                    │
│                              │   execution/broker_alpaca.py  │                                    │
│                              │   @require_no_kill_switch     │                                    │
│                              ├───────────────────────────────┤                                    │
│                              │ • place_ioc_limit()           │                                    │
│                              │ • limit = best_ask × 1.001    │                                    │
│                              │ • IOC (Immediate-Or-Cancel)   │                                    │
│                              │ • execute_signal()            │                                    │
│                              └───────────────┬───────────────┘                                    │
│                                              │                                                    │
│                                              ▼                                                    │
│                              ┌───────────────────────────────┐                                    │
│                              │         ALPACA API            │                                    │
│                              │    (Paper or Live mode)       │                                    │
│                              └───────────────────────────────┘                                    │
│                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────── MONITORING & SELF-HEALING ────────────────────────────────┐
│                                                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Anomaly Detect  │  │ Drift Detector  │  │ Health Endpoint │  │ Hash Chain      │              │
│  │ selfmonitor/    │  │ monitor/        │  │ monitor/        │  │ core/           │              │
│  │ anomaly_detect  │  │ drift_detector  │  │ health_endpts   │  │ hash_chain.py   │              │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤              │
│  │ • Z-score price │  │ • Rolling WR/PF │  │ • /health       │  │ • SHA256 chain  │              │
│  │ • Volume spikes │  │ • Sharpe track  │  │ • /ready        │  │ • Tamper detect │              │
│  │ • Auto-alert    │  │ • Standdown rec │  │ • /metrics      │  │ • Audit verify  │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────── EVOLUTION & OPTIMIZATION ─────────────────────────────────┐
│                                                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Genetic Optim   │  │ Strategy Mutate │  │ Rule Generator  │  │ Promotion Gate  │              │
│  │ evolution/      │  │ evolution/      │  │ evolution/      │  │ evolution/      │              │
│  │ genetic_optim   │  │ strategy_mutate │  │ rule_generator  │  │ promotion_gate  │              │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤              │
│  │ • Tournament    │  │ • Param perturb │  │ • Rule templates│  │ • OOS Sharpe>1  │              │
│  │ • Crossover     │  │ • Gaussian noise│  │ • Auto-generate │  │ • PF>1.5        │              │
│  │ • Mutation      │  │ • Bounds check  │  │ • Entry/exit    │  │ • Min 30 trades │              │
│  │ • Elitism       │  │ • Clone protect │  │ • Stop/target   │  │ • WF validation │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────── TESTING & STRESS ─────────────────────────────────────────┐
│                                                                                                   │
│  ┌─────────────────────────────────┐      ┌─────────────────────────────────┐                    │
│  │        MONTE CARLO              │      │        STRESS TESTING           │                    │
│  │      testing/monte_carlo.py     │      │      testing/stress_test.py     │                    │
│  ├─────────────────────────────────┤      ├─────────────────────────────────┤                    │
│  │ • 10,000 simulations            │      │ • Black Monday (-22%)           │                    │
│  │ • VaR (95%)                     │      │ • COVID Crash (-34%)            │                    │
│  │ • CVaR (Expected Shortfall)     │      │ • VIX Spike (2x vol)            │                    │
│  │ • Max Drawdown distribution     │      │ • Flash Crash                   │                    │
│  │ • Probability of profit         │      │ • Survival analysis             │                    │
│  └─────────────────────────────────┘      └─────────────────────────────────┘                    │
│                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────── EXPLAINABILITY & REPORTS ─────────────────────────────────┐
│                                                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Trade Explainer │  │ Narrative Gen   │  │ Decision Track  │  │ Reports Output  │              │
│  │ explainability/ │  │ explainability/ │  │ explainability/ │  │                 │              │
│  │ trade_explainer │  │ narrative_gen   │  │ decision_track  │  │ reports/        │              │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤              │
│  │ • Why this trade│  │ • Technical     │  │ • Full audit    │  │ • Morning HTML  │              │
│  │ • Factor contrib│  │ • Casual        │  │ • Every decision│  │ • EOD summary   │              │
│  │ • Human readable│  │ • Executive     │  │ • Timestamps    │  │ • Top-3 picks   │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────── OPTIONS ENGINE ───────────────────────────────────────────┐
│                                                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Volatility Est  │  │ Strike Select   │  │ Position Sizing │  │ Greeks Calc     │              │
│  │ options/        │  │ options/        │  │ options/        │  │ options/        │              │
│  │ volatility.py   │  │ selection.py    │  │ position_sizing │  │ pricing.py      │              │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤              │
│  │ • Close-to-close│  │ • Delta target  │  │ • 2% risk/trade │  │ • Black-Scholes │              │
│  │ • Parkinson     │  │ • Binary search │  │ • Contract size │  │ • Delta/Gamma   │              │
│  │ • Yang-Zhang    │  │ • OTM/ITM       │  │ • Premium calc  │  │ • Theta/Vega    │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────── 24/7 SCHEDULER ───────────────────────────────────────────┐
│                                                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              scripts/runner.py                                             │   │
│  │                                                                                            │   │
│  │   ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐         │   │
│  │   │ 09:35   │ ───► │ 10:30   │ ───► │ 15:55   │ ───► │  EOD    │ ───► │ Next    │         │   │
│  │   │ Scan #1 │      │ Scan #2 │      │ Scan #3 │      │ Recon   │      │  Day    │         │   │
│  │   └─────────┘      └─────────┘      └─────────┘      └─────────┘      └─────────┘         │   │
│  │                                                                                            │   │
│  │   Mode: --mode paper  or  --mode live                                                      │   │
│  │   Universe: 900 stocks, Cap: configurable                                                  │   │
│  │                                                                                            │   │
│  └───────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Layer Summary

| Layer | Description |
|-------|-------------|
| **DATA LAYER** | Ingests EOD OHLCV from Polygon/Stooq/Yahoo. 900-stock universe with 10yr history. CSV caching for speed. Data quality checks before any processing. |
| **RESEARCH LAYER** | 25 features (momentum, vol, trend, technical) + 18 alphas with economic hypotheses. Walk-forward screener ranks alphas. Feature importance analysis. |
| **STRATEGY LAYER** | Two strategies: Donchian (trend breakout) + ICT Turtle Soup (mean reversion). Both use SMA(200) filter, Wilder ATR(14), 2x ATR stop, 5-bar time exit. |
| **ML/META-MODEL** | Ensemble combines strategy signals. Confidence gate (0.6 threshold) filters. Brier score calibration. Drift detection for degradation alerts. |
| **BACKTEST ENGINE** | Walk-forward validation (252-train/63-test). FIFO P&L. Experiment registry for reproducibility. Options backtesting with synthetic Black-Scholes. |
| **RISK MANAGEMENT** | PolicyGate ($75/order, $1k/day). LiquidityGate (ADV, spread). Circuit breaker for consecutive losses. Kill switch for emergency halt. |
| **COMPLIANCE** | Rules engine (position size, PDT, penny stocks). Prohibited list with expiration. Hash-verified audit trail for every action. |
| **EXECUTION** | OMS with idempotency (no duplicate orders). IOC LIMIT orders via Alpaca. Kill switch decorator blocks all orders when activated. |
| **MONITORING** | Anomaly detection (Z-score). Drift detector (WR/PF/Sharpe). Health endpoints. Hash chain for tamper-proof audit logs. |
| **EVOLUTION** | Genetic optimizer for parameter search. Strategy mutation. Promotion gates require OOS Sharpe>1, PF>1.5, min 30 trades before production. |
| **TESTING/STRESS** | Monte Carlo (10k sims, VaR, CVaR). Stress scenarios (Black Monday, COVID, VIX). Survival analysis for drawdown tolerance. |
| **EXPLAINABILITY** | Trade explainer (why this trade). Narrative generator (technical/casual/exec). Decision tracker logs every decision with timestamps. |
| **OPTIONS ENGINE** | Synthetic Black-Scholes pricing. Delta-targeted strike selection. Volatility estimation (Parkinson, Yang-Zhang). 2% risk position sizing. |
| **24/7 SCHEDULER** | Runs scans at 09:35, 10:30, 15:55. EOD reconciliation. Paper or live mode. State persistence across restarts. |

## Quick Stats

- **Tests:** 533 passing
- **CI:** Python 3.11 & 3.12
- **Modules:** 17 readiness items
- **Skills:** 70 slash commands
