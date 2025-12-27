Run Kobe 24/7 (Windows)

Overview
- A simple always-on runner triggers submissions at set times (local): `scripts/runner.py`.
- Use Windows Task Scheduler or NSSM to run it on boot.
- Safety: kill switch (`state/KILL_SWITCH`), budgets via PolicyGate, structured logs, audit chain.

Runner Options
- Paper micro (default):
  python scripts/runner.py --mode paper --universe data/universe/optionable_liquid_900.csv --cap 50 --scan-times 09:35,10:30,15:55 --lookback-days 540 --dotenv ./.env

- Live micro:
  Ensure `.env` points `ALPACA_BASE_URL` to live, then:
  python scripts/runner.py --mode live --universe data/universe/optionable_liquid_900.csv --cap 10 --scan-times 09:35,10:30,15:55 --lookback-days 540 --dotenv ./.env

Task Scheduler Setup (quick)
1) Open Task Scheduler â†’ Create Taskâ€¦
2) General: Run whether user is logged on or not; Run with highest privileges.
3) Triggers: At startup (or at 8:55AM every weekday) â†’ Enabled.
4) Actions: Start a program:
   Program/script: C:\\Windows\\System32\\cmd.exe
   Add arguments: /c "cd /d C:\\Users\\Owner\\OneDrive\\Desktop\\kobe81_traderbot && C:\\Users\\Owner\\AppData\\Local\\Programs\\Python\\Python311\\python.exe scripts\\runner.py --mode paper --universe data\\universe\\optionable_liquid_900.csv --cap 50 --scan-times 09:35,10:30,15:55 --lookback-days 540 --dotenv C:\\Users\\Owner\\OneDrive\\Desktop\\.\\.env"
5) Conditions: Uncheck "Start the task only if the computer is on AC power" if desired.
6) Settings: Restart on failure; If the task is already running, do not start a new instance.

Log & Audit
- Structured logs: `logs/events.jsonl`
- Order audit chain: `state/hash_chain.jsonl` (verify with `scripts/verify_hash_chain.py`)

Health & Reconciliation
- Health: python scripts/start_health.py --port 8000
- Reconcile snapshots: python scripts/reconcile_alpaca.py --dotenv ./.env



