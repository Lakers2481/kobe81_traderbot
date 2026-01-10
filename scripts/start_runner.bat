@echo off
REM ============================================================
REM Kobe Trading System - Runner Startup Script
REM ============================================================
REM This script starts the 24/7 trading runner.
REM Use Windows Task Scheduler to run this at system startup.
REM ============================================================

cd /d "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot"

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Set trading mode (paper or live)
set MODE=paper

REM Run the 24/7 runner
REM Scan times: 10:00 (primary), 10:30 (fallback), 14:30 (power hour), 15:30 (EOD)
python scripts/runner.py ^
    --mode %MODE% ^
    --universe data/universe/optionable_liquid_800.csv ^
    --cap 900 ^
    --scan-times 10:00,10:30,14:30,15:30 ^
    --dotenv .env

REM If runner exits, log it
echo Runner exited at %date% %time% >> logs\runner_exits.log
