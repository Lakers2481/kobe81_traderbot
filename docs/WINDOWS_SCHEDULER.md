Windows Task Scheduler â€” Kobe Master Scheduler

Import the task:
1) Open Task Scheduler
2) Action > Import Task...
3) Select: ops/windows/kobe_scheduler_task.xml
4) Confirm the Working Directory points to your Kobe repo path. Default:
   C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot
5) Ensure python is on PATH. If not, edit Action to full python.exe path.

What it does
- Triggers on system startup and at user logon
- Runs: python scripts\scheduler_kobe.py --dotenv .\.env --universe data\universe\optionable_liquid_900.csv --cap 900 --min-conf 0.60 --tick-seconds 20 --telegram
- The scheduler stays running and executes the full daily plan in ET.

Telegram
- Set in .env or 2K28 .env fallback (scheduler loads TELEGRAM_* from C:\Users\Owner\OneDrive\Desktop\GAME_PLAN_2K28\.env if present).
- Required variables:
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_CHAT_ID=...
  TELEGRAM_ALERTS_ENABLED=true


