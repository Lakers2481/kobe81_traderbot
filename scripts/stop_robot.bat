@echo off
REM ============================================================
REM KOBE TRADING SYSTEM - STOP SCRIPT
REM ============================================================
REM This script gracefully stops the Kobe autonomous brain.
REM ============================================================

echo ============================================================
echo KOBE AUTONOMOUS TRADING SYSTEM - STOP
echo ============================================================
echo.

REM Set working directory to script location
cd /d "%~dp0\.."

REM Check if PID file exists
if not exist "state\autonomous\kobe.pid" (
    echo No running instance found (no PID file)
    echo.
    pause
    exit /b 0
)

echo Sending stop signal...
python -m autonomous.run --stop

if errorlevel 1 (
    echo.
    echo WARNING: Stop command returned error
    echo The brain may not have stopped cleanly
    echo.
    choice /c YN /m "Do you want to force kill"
    if errorlevel 2 exit /b 0

    REM Force kill by removing PID file
    del "state\autonomous\kobe.pid" 2>nul
    echo PID file removed
)

echo.
echo ============================================================
echo KOBE BRAIN STOPPED
echo ============================================================

pause
