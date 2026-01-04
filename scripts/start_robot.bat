@echo off
REM ============================================================
REM KOBE TRADING SYSTEM - START SCRIPT
REM ============================================================
REM This script starts the Kobe autonomous brain in 24/7 mode.
REM Run this script to begin autonomous trading operations.
REM ============================================================

echo ============================================================
echo KOBE AUTONOMOUS TRADING SYSTEM
echo ============================================================
echo.

REM Set working directory to script location
cd /d "%~dp0\.."

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.11+ and add to PATH
    pause
    exit /b 1
)

REM Check if already running
if exist "state\autonomous\kobe.pid" (
    echo WARNING: Kobe may already be running
    echo PID file exists: state\autonomous\kobe.pid
    echo.
    choice /c YN /m "Do you want to continue anyway"
    if errorlevel 2 exit /b 0
)

REM Run preflight checks
echo Running preflight checks...
python scripts\preflight.py --dotenv .\.env
if errorlevel 1 (
    echo.
    echo ERROR: Preflight checks failed
    echo Please fix the issues above and try again
    pause
    exit /b 1
)

echo.
echo ============================================================
echo STARTING KOBE BRAIN (24/7 Mode)
echo ============================================================
echo.
echo Press Ctrl+C to stop gracefully
echo.

REM Start the brain
python -m autonomous.run --start

echo.
echo ============================================================
echo KOBE BRAIN STOPPED
echo ============================================================

pause
