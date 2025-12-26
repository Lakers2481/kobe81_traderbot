Kobe81 Traderbot — Backtesting, Paper, Live (Micro)

Overview
- Strategies: canonical Connors RSI‑2 and IBS (with AND filter).
- Data: Polygon daily OHLCV with caching.
- Universe: optionable + liquid candidates filtered to final 950 with ≥10y coverage.
- Backtesting: deterministic next‑bar fills, ATR(14)×2 stop, 5‑bar time stop, no lookahead.
- Outputs: trade_list.csv, equity_curve.csv, summary.json per run/split.
- Walk‑forward: rolling splits, side‑by‑side RSI‑2 vs IBS vs AND, HTML report.
- Execution: Alpaca IOC LIMIT submission (paper or live), kill switch, budgets, idempotency, audit log.

Project Map
- strategies/ — RSI‑2 and IBS implementations
- backtest/ — engine + walk‑forward
- data/ — providers (Polygon) + universe loader
- execution/ — Alpaca broker adapter (IOC limit)
- risk/ — PolicyGate (budgets, bounds)
- oms/ — order state + idempotency store
- core/ — hash‑chain audit, config pin, structured logs
- monitor/ — health endpoints
- scripts/ — preflight, build/prefetch, WF/report, showdown, paper/live, reconcile, runner
- docs/ — COMPLETE_ROBOT_ARCHITECTURE.md, RUN_24x7.md, docs index

Requirements
- Python 3.11+
- Install: `pip install -r requirements.txt`
- Env: set in `C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env`
  - `POLYGON_API_KEY=...`
  - `ALPACA_API_KEY_ID=...`
  - `ALPACA_API_SECRET_KEY=...`
  - `ALPACA_BASE_URL=https://paper-api.alpaca.markets` (paper) or live endpoint

Quick Start
1) Preflight (keys + config pin + broker probe)
   python scripts/preflight.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

2) Build the 950‑stock universe with proofs
   python scripts/build_universe_polygon.py --candidates data/universe/optionable_liquid_candidates.csv --start 2015-01-01 --end 2024-12-31 --min-years 10 --cap 950 --concurrency 3 --cache data/cache --out data/universe/optionable_liquid_final.csv --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

3) Prefetch EOD bars (faster WF)
   python scripts/prefetch_polygon_universe.py --universe data/universe/optionable_liquid_final.csv --start 2015-01-01 --end 2024-12-31 --cache data/cache --concurrency 3 --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

4) Walk‑forward comparison and report
   python scripts/run_wf_polygon.py --universe data/universe/optionable_liquid_final.csv --start 2015-01-01 --end 2024-12-31 --train-days 252 --test-days 63 --cap 950 --outdir wf_outputs --cache data/cache --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
   python scripts/aggregate_wf_report.py --wfdir wf_outputs --out wf_outputs/wf_report.html

5) Showdown (full‑period side‑by‑side)
   python scripts/run_showdown_polygon.py --universe data/universe/optionable_liquid_final.csv --start 2015-01-01 --end 2024-12-31 --cap 950 --outdir showdown_outputs --cache data/cache --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

Paper and Live Trading (IOC LIMIT)
- Paper (micro budgets):
  python scripts/run_paper_trade.py --universe data/universe/optionable_liquid_final.csv --start 2015-01-01 --end 2024-12-31 --cap 50 --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

- Live (micro budgets; set ALPACA_BASE_URL to live in .env):
  python scripts/run_live_trade_micro.py --universe data/universe/optionable_liquid_final.csv --start 2015-01-01 --end 2024-12-31 --cap 10 --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

Evidence Artifacts
- wf_outputs/wf_summary_compare.csv — strategy side‑by‑side KPIs
- wf_outputs/<strategy>/split_NN/{trade_list.csv,equity_curve.csv,summary.json}
- showdown_outputs/showdown_summary.csv, showdown_report.html
- data/universe/optionable_liquid_final.csv and `.full.csv` (coverage, ADV, options proofs)
- logs/events.jsonl (structured), state/hash_chain.jsonl (audit)

24/7 Runner
- Paper example:
  python scripts/runner.py --mode paper --universe data/universe/optionable_liquid_final.csv --cap 50 --scan-times 09:35,10:30,15:55 --lookback-days 540 --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
- Live example:
  python scripts/runner.py --mode live --universe data/universe/optionable_liquid_final.csv --cap 10 --scan-times 09:35,10:30,15:55 --lookback-days 540 --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
- Task Scheduler setup: see docs/RUN_24x7.md

Safety
- Kill switch: create file `state/KILL_SWITCH` to halt submissions.
- Policy Gate: per‑order and daily budgets ($75 / $1,000), price bounds, shorts disabled by default.
- Audit: verify `python scripts/verify_hash_chain.py`.
- Reconciliation: `python scripts/reconcile_alpaca.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env`.

