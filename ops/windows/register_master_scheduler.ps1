# Register Kobe Master Scheduler with Windows Task Scheduler
#
# This script registers the scheduler_kobe.py daemon to run at logon/boot.
# The master scheduler handles all 40+ daily jobs in a single process.
#
# REQUIRES: Administrator privileges (Run as Administrator)
#
# Usage:
#   # Right-click PowerShell -> Run as Administrator, then:
#   powershell -ExecutionPolicy Bypass -File ops\windows\register_master_scheduler.ps1
#
# Options:
#   -Unregister    Remove the task instead of registering it
#   -Start         Start the task immediately after registration
#   -Status        Show task status and exit
#   -CurrentUser   Register for current user only (no admin required)
#   -WithWatchdog  Also register the 5-minute watchdog task

param(
    [switch]$Unregister,
    [switch]$Start,
    [switch]$Status,
    [switch]$CurrentUser,
    [switch]$WithWatchdog
)

# Check if running as admin (unless -CurrentUser is specified)
function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal $identity
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not $Status -and -not $CurrentUser -and -not (Test-Admin)) {
    Write-Host ""
    Write-Host "ERROR: Administrator privileges required!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  1. Right-click PowerShell -> 'Run as Administrator'"
    Write-Host "  2. Use -CurrentUser flag to register for current user only"
    Write-Host ""
    Write-Host "Example (current user, no admin):"
    Write-Host "  powershell -ExecutionPolicy Bypass -File ops\windows\register_master_scheduler.ps1 -CurrentUser"
    Write-Host ""
    exit 1
}

$TaskName = "Kobe_Master_Scheduler"
$TaskXml = Join-Path $PSScriptRoot "kobe_scheduler_task.xml"

# Show status if requested
if ($Status) {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($task) {
        Write-Host "Task: $TaskName"
        Write-Host "State: $($task.State)"
        $info = Get-ScheduledTaskInfo -TaskName $TaskName
        Write-Host "Last Run: $($info.LastRunTime)"
        Write-Host "Next Run: $($info.NextRunTime)"
        Write-Host "Last Result: $($info.LastTaskResult)"
    } else {
        Write-Host "Task '$TaskName' is not registered."
    }
    exit 0
}

# Unregister if requested
if ($Unregister) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Task '$TaskName' unregistered."
    } else {
        Write-Host "Task '$TaskName' was not registered."
    }
    exit 0
}

# Check if XML exists
if (-not (Test-Path $TaskXml)) {
    Write-Error "Task XML not found: $TaskXml"
    exit 1
}

# Check if task exists
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($existing) {
    Write-Host "Task '$TaskName' already exists. Updating..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Register the task
try {
    if ($CurrentUser) {
        # Register for current user only (no admin required)
        $RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
        $action = New-ScheduledTaskAction -Execute "python" `
            -Argument "scripts\scheduler_kobe.py --dotenv .\.env --universe data\universe\optionable_liquid_800.csv --cap 900 --tick-seconds 20 --telegram" `
            -WorkingDirectory $RepoRoot
        $triggerLogon = New-ScheduledTaskTrigger -AtLogon
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
        Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggerLogon -Settings $settings | Out-Null
    } else {
        # Use the XML file (requires admin for boot trigger)
        Register-ScheduledTask -Xml (Get-Content $TaskXml -Raw) -TaskName $TaskName | Out-Null
    }
    Write-Host ""
    Write-Host "====================================="
    Write-Host " Kobe Master Scheduler Registered"
    Write-Host "====================================="
    Write-Host ""
    Write-Host "Task Name: $TaskName"
    Write-Host "Triggers:  At user logon, At system boot"
    Write-Host "Action:    python scripts\scheduler_kobe.py --telegram"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  Start:    Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host "  Stop:     Stop-ScheduledTask -TaskName '$TaskName'"
    Write-Host "  Status:   .\register_master_scheduler.ps1 -Status"
    Write-Host "  Remove:   .\register_master_scheduler.ps1 -Unregister"
    Write-Host ""
} catch {
    Write-Error "Failed to register task: $_"
    exit 1
}

# Start immediately if requested
if ($Start) {
    Write-Host "Starting task..."
    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep -Seconds 2
    $task = Get-ScheduledTask -TaskName $TaskName
    Write-Host "Task state: $($task.State)"
}

# Register watchdog task if requested
if ($WithWatchdog) {
    $WatchdogName = "Kobe_Watchdog"
    $WatchdogXml = Join-Path $PSScriptRoot "kobe_watchdog_task.xml"

    if (-not (Test-Path $WatchdogXml)) {
        Write-Warning "Watchdog XML not found: $WatchdogXml"
    } else {
        # Remove existing watchdog task if present
        $existingWatchdog = Get-ScheduledTask -TaskName $WatchdogName -ErrorAction SilentlyContinue
        if ($existingWatchdog) {
            Unregister-ScheduledTask -TaskName $WatchdogName -Confirm:$false
        }

        try {
            Register-ScheduledTask -Xml (Get-Content $WatchdogXml -Raw) -TaskName $WatchdogName | Out-Null
            Write-Host ""
            Write-Host "====================================="
            Write-Host " Kobe Watchdog Registered"
            Write-Host "====================================="
            Write-Host ""
            Write-Host "Task Name: $WatchdogName"
            Write-Host "Trigger:   Every 5 minutes"
            Write-Host "Action:    python scripts\watchdog.py --restart-if-dead"
            Write-Host ""
            Write-Host "The watchdog will automatically restart the scheduler if it crashes."
            Write-Host ""
        } catch {
            Write-Warning "Failed to register watchdog task: $_"
        }
    }
}
