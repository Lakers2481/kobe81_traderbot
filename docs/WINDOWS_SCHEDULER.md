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
- Runs: python scripts\\scheduler_kobe.py --dotenv .\\.env --universe data\\universe\\optionable_liquid_800.csv --cap 900 --min-conf 0.60 --tick-seconds 20 --telegram
- The scheduler stays running and executes the full daily plan in ET.

Telegram
- Set in .env or 2K28 .env fallback (scheduler loads TELEGRAM_* from C:\Users\Owner\OneDrive\Desktop\GAME_PLAN_2K28\.env if present).
- Required variables:
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_CHAT_ID=...
  TELEGRAM_ALERTS_ENABLED=true

Per‑job task Telegram (optional)
- Use the registrar script with `-PerJobTelegram` to receive a Telegram message per job (MORNING_REPORT, FIRST_SCAN, etc.):
  powershell -ExecutionPolicy Bypass -File ops\windows\register_all_tasks.ps1 -PerJobTelegram -HeartbeatTelegram
  This passes `--telegram --telegram-dotenv C:\Users\Owner\OneDrive\Desktop\GAME_PLAN_2K28\.env` to scripts/run_job.py for all tasks.



Time Zones
---------
- Trading operations (job schedule, scan dates) run on New York time (ET).
- Displays (live dashboard) show Central Time (CT) by default for your location.
- Windows Task registration uses ET times converted to your local clock, so jobs run at the intended ET hours.

Task Time Cheat Sheet (ET → CT)
--------------------------------
ET is typically 1 hour ahead of CT. The register_all_tasks.ps1 script converts these ET times to your local time automatically.

- 5:30 AM ET  4:30 AM CT - DB_BACKUP — DB_BACKUP
- 6:00 AM ET  5:00 AM CT - DATA_UPDATE (cache warm) — DATA_UPDATE (cache warm)
- 6:30 AM ET  5:30 AM CT - MORNING_REPORT — MORNING_REPORT
- 6:45 AM ET  5:45 AM CT - MORNING_CHECK — MORNING_CHECK
- 8:00 AM ET  7:00 AM CT - PRE_GAME (sentiment) — PRE_GAME (sentiment)
- 9:00 AM ET  8:00 AM CT - MARKET_NEWS (sentiment refresh) — MARKET_NEWS (sentiment refresh)
- 9:15 AM ET  8:15 AM CT - PREMARKET_SCAN (plan) — PREMARKET_SCAN (plan)
- 9:45 AM ET  8:45 AM CT - FIRST_SCAN (Top-3 + TOTD) — FIRST_SCAN (Top‑3 + TOTD)
- 12:00 PM ET  11:00 AM CT - HALF_TIME — HALF_TIME
- 2:30 PM ET  1:30 PM CT - AFTERNOON_SCAN — AFTERNOON_SCAN
- 3:30 PM ET  2:30 PM CT - SWING_SCANNER — SWING_SCANNER
- 4:00 PM ET  3:00 PM CT - POST_GAME — POST_GAME
- 4:05 PM ET  3:05 PM CT - EOD_REPORT — EOD_REPORT
- 5:00 PM ET  4:00 PM CT - EOD_LEARNING (Fridays) — EOD_LEARNING (Fridays)
- 9:00 PM ET  8:00 PM CT - OVERNIGHT_ANALYSIS — OVERNIGHT_ANALYSIS

Notes
- Daylight saving time is handled by Windows. The registration script converts ET → local using the OS timezone tables.
- The live console dashboard stamps the header and deadlines in CT and labels them “(CT)”.



- Optional: STATUS watcher at logon — import ops/windows/watch_status_task.xml in Task Scheduler to auto-start ops/windows/watch_status.ps1 at user logon so docs/STATUS.md updates on code changes.

