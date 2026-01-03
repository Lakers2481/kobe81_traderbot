Kobe81 Traderbot — Interview One‑Pager

What it is
- Canonical daily U.S. equities system with two mean‑reversion strategies: IBS+RSI and ICT Turtle Soup.
- Deterministic backtester (next‑bar fills), ATR stops and time stops; no lookahead.
- Evidence artifacts auto‑generated (CSV + HTML). The quick test completes in minutes.

Why it’s valid
- In‑sample (2015–2022): ~60–61% win rate, ~1.44 profit factor (see README and `backtest_dual_strategy.py`).
- Out‑of‑sample (2023–2024): ≥60% win rate, ~1.60 profit factor on unseen data.
- Controls: sweep‑strength filter, regime filter (SPY SMA/vol gate), optional commissions/slippage model.

How to run (3 commands)
1) Install deps: `pip install -r requirements.txt`
2) Set keys in `./.env` (`POLYGON_API_KEY`, `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`)
3) Quick test + report: `python scripts/interview_quick_test.py --dotenv ./.env`

Artifacts
- HTML report: `wf_outputs_interview_quick/wf_report.html`
- KPIs CSV: `wf_outputs_interview_quick/wf_summary_compare.csv`
- Summary JSON: `INTERVIEW_SUMMARY.json`

Metrics to discuss
- Win rate, Profit factor, Max drawdown per strategy
- Walk‑forward split averages (train/test cadence)
- Commission model impact (enable in `config/base.yaml`)

Suggested demo flow (5–7 minutes)
1) Preflight: `python scripts/preflight.py --dotenv ./.env`
2) Run quick WF: `python scripts/interview_quick_test.py --dotenv ./.env`
3) Open report and summarize WR/PF vs. in‑sample/OOS expectations
4) If time: re‑run with commissions enabled or show Monte Carlo robustness (`scripts/monte_carlo.py`)

Safety & operations
- Kill switch: `state/KILL_SWITCH` file halts submissions.
- Risk budgets and limits: `risk/policy_gate.py`, `config/base.yaml` (modes: micro/paper/real).
- Audit and reconciliation: hash‑chain (`scripts/verify_hash_chain.py`) and Alpaca reconciliation.

