Kobe81 Traderbot â€” Interview Oneâ€‘Pager

What it is
- Canonical daily equities system with Connors RSIâ€‘2, ICT Turtle Soup, ,  TOPâ€‘N selector.
- Deterministic backtester (nextâ€‘bar fills), ATR stops  time stops; no lookahead.
- Evidence artifacts autoâ€‘generated (CSV + HTML); quick test finishes in minutes.

How to run (3 comms)
1) Install deps: `pip install -r requirements.txt`
2) Set keys in `./.env` (POLYGON_API_KEY, ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY)
3) Quick test + report:
   `python scripts/interview_quick_test.py --dotenv ./.env`

Artifacts
- Report: `wf_outputs_interview_quick/wf_report.html`
- KPIs CSV: `wf_outputs_interview_quick/wf_summary_compare.csv`
- Summary JSON: `INTERVIEW_SUMMARY.json`

Metrics to discuss
- Win rate, Profit factor, Max drawdown (per strategy row)
- Walkâ€‘forward split averages (train/test cadence)
- Commission model optional (enable in `config/base.yaml`)

Notes
- Universe: `data/universe/optionable_liquid_final.csv` (cap=50 in quick run)
- Data: Polygon daily OHLCV with CSV caching under `data/cache/`
- Safety: kill switch `state/KILL_SWITCH`, risk budgets in `risk/policy_gate.py`


