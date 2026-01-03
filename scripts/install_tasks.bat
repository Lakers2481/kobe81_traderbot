@echo off
REM ============================================================
REM Install Kobe Trading System Scheduled Tasks
REM Run this script as Administrator
REM ============================================================

echo ============================================================
echo Kobe Trading System - Task Scheduler Setup
echo ============================================================

set PROJECT=C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot

REM Delete existing tasks if they exist
schtasks /Delete /TN "KobeRunner" /F 2>nul
schtasks /Delete /TN "KobeHealthCheck" /F 2>nul
schtasks /Delete /TN "KobeExitManager" /F 2>nul

echo.
echo [1/3] Creating KobeRunner task...
schtasks /Create /TN "KobeRunner" /TR "cmd.exe /c \"%PROJECT%\scripts\start_runner.bat\"" /SC DAILY /ST 09:00 /RL HIGHEST /F
if %ERRORLEVEL% EQU 0 (
    echo   SUCCESS: KobeRunner task created
) else (
    echo   FAILED: Could not create KobeRunner task
)

echo.
echo [2/3] Creating KobeHealthCheck task...
schtasks /Create /TN "KobeHealthCheck" /TR "python \"%PROJECT%\scripts\health_monitor.py\" --check-runner --alert-on-failure" /SC MINUTE /MO 5 /ST 09:00 /ET 17:00 /F
if %ERRORLEVEL% EQU 0 (
    echo   SUCCESS: KobeHealthCheck task created
) else (
    echo   FAILED: Could not create KobeHealthCheck task
)

echo.
echo [3/3] Creating KobeExitManager task...
schtasks /Create /TN "KobeExitManager" /TR "python \"%PROJECT%\scripts\exit_manager.py\" --check-time-exits --execute" /SC MINUTE /MO 30 /ST 10:00 /ET 17:00 /F
if %ERRORLEVEL% EQU 0 (
    echo   SUCCESS: KobeExitManager task created
) else (
    echo   FAILED: Could not create KobeExitManager task
)

echo.
echo ============================================================
echo Setup complete! Listing Kobe tasks:
echo ============================================================
schtasks /Query /TN "KobeRunner" /FO LIST 2>nul
schtasks /Query /TN "KobeHealthCheck" /FO LIST 2>nul
schtasks /Query /TN "KobeExitManager" /FO LIST 2>nul

echo.
echo To start the runner immediately:
echo   schtasks /Run /TN "KobeRunner"
echo.
