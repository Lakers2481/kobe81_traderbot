param(
  [string]$RepoRoot = "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot",
  [string]$PythonExe = "python",
  [string]$DotEnv = ".\\.env",
  [string]$Universe = "data\universe\optionable_liquid_900.csv",
  [int]$Cap = 900,
  [double]$MinConf = 0.60,
  [switch]$HeartbeatTelegram,
  [switch]$PerJobTelegram,
  [string]$TelegramDotenv = 'C:\Users\Owner\OneDrive\Desktop\GAME_PLAN_2K28\.env',
  [string]$EtTimeZoneId = 'Eastern Standard Time'
)

$ErrorActionPreference = 'Stop'

function New-KobeJobTask {
  param(
    [string]$Name,
    [string]$Tag,
    [datetime]$At,
    [string[]]$DaysOfWeek = @('Monday','Tuesday','Wednesday','Thursday','Friday')
  )
  $args = "/c cd /d `"$RepoRoot`" && $PythonExe scripts\run_job.py --tag $Tag --dotenv $DotEnv --universe $Universe --cap $Cap --min-conf $MinConf"
  if ($PerJobTelegram) { $args += " --telegram --telegram-dotenv `"$TelegramDotenv`"" }
  $action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument $args
  $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $DaysOfWeek -At $At
  Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Force | Out-Null
  Write-Host "Registered task: $Name ($Tag at $($At.ToShortTimeString()))"
}

# Convert an ET clock time (hour/min) to local DateTime for today (handles DST)
function Get-LocalFromET([int]$HourEt, [int]$MinuteEt) {
  try { $et = [System.TimeZoneInfo]::FindSystemTimeZoneById($EtTimeZoneId) } catch { $et = [System.TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time') }
  $local = [System.TimeZoneInfo]::Local
  $todayLocal = (Get-Date).Date
  $etUnspec = New-Object datetime ($todayLocal.Year, $todayLocal.Month, $todayLocal.Day, $HourEt, $MinuteEt, 0)
  return [System.TimeZoneInfo]::ConvertTime($etUnspec, $et, $local)
}

# Register a job using ET schedule converted to local clock
function New-KobeEtJobTask {
  param(
    [string]$Name,
    [string]$Tag,
    [int]$HourEt,
    [int]$MinEt,
    [string[]]$DaysOfWeek = @('Monday','Tuesday','Wednesday','Thursday','Friday')
  )
  $atLocal = Get-LocalFromET -HourEt $HourEt -MinuteEt $MinEt
  New-KobeJobTask -Name $Name -Tag $Tag -At $atLocal -DaysOfWeek $DaysOfWeek
}

# Heartbeat (every 1 minute)
function New-KobeHeartbeatTask {
  param([string]$Name = 'Kobe_HEARTBEAT')
  $hbArgs = "/c cd /d `"$RepoRoot`" && $PythonExe scripts\heartbeat.py --dotenv $DotEnv"
  if ($HeartbeatTelegram) { $hbArgs += " --telegram" }
  $action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument $hbArgs
  $start = Get-Date ((Get-Date).Date)  # today midnight local
  $trigger = New-ScheduledTaskTrigger -Once -At $start -RepetitionInterval (New-TimeSpan -Minutes 1) -RepetitionDuration (New-TimeSpan -Days 3650)
  Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Force | Out-Null
  Write-Host "Registered task: $Name (every 1 minute)"
}

# Repeating supervisor task (every N minutes)
function New-KobeSupervisorTask {
  param([string]$Name = 'Kobe_SUPERVISOR',[int]$EveryMinutes = 10)
  $args = "/c cd /d `"$RepoRoot`" && $PythonExe scripts\supervisor.py"
  $action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument $args
  $start = Get-Date ((Get-Date).AddMinutes(1))
  $trigger = New-ScheduledTaskTrigger -Once -At $start -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes) -RepetitionDuration (New-TimeSpan -Days 3650)
  Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Force | Out-Null
  Write-Host "Registered task: $Name (every $EveryMinutes min)"
}

# Register all discrete daily jobs at ET times (converted to local)
New-KobeEtJobTask -Name 'Kobe_DB_BACKUP'      -Tag 'DB_BACKUP'          -HourEt 5  -MinEt 30
New-KobeEtJobTask -Name 'Kobe_DATA_UPDATE'    -Tag 'DATA_UPDATE'        -HourEt 6  -MinEt 0
New-KobeEtJobTask -Name 'Kobe_MORNING_REPORT' -Tag 'MORNING_REPORT'     -HourEt 6  -MinEt 30
New-KobeEtJobTask -Name 'Kobe_MORNING_CHECK'  -Tag 'MORNING_CHECK'      -HourEt 6  -MinEt 45
New-KobeEtJobTask -Name 'Kobe_PRE_GAME'       -Tag 'PRE_GAME'           -HourEt 8  -MinEt 0
New-KobeEtJobTask -Name 'Kobe_MARKET_NEWS'    -Tag 'MARKET_NEWS'        -HourEt 9  -MinEt 0
New-KobeEtJobTask -Name 'Kobe_PREMARKET_SCAN' -Tag 'PREMARKET_SCAN'     -HourEt 9  -MinEt 15
New-KobeEtJobTask -Name 'Kobe_FIRST_SCAN'     -Tag 'FIRST_SCAN'         -HourEt 9  -MinEt 45
New-KobeEtJobTask -Name 'Kobe_HALF_TIME'      -Tag 'HALF_TIME'          -HourEt 12 -MinEt 0
New-KobeEtJobTask -Name 'Kobe_AFTERNOON_SCAN' -Tag 'AFTERNOON_SCAN'     -HourEt 14 -MinEt 30
New-KobeEtJobTask -Name 'Kobe_SWING_SCANNER'  -Tag 'SWING_SCANNER'      -HourEt 15 -MinEt 30
New-KobeEtJobTask -Name 'Kobe_POST_GAME'      -Tag 'POST_GAME'          -HourEt 16 -MinEt 0
New-KobeEtJobTask -Name 'Kobe_EOD_REPORT'     -Tag 'EOD_REPORT'         -HourEt 16 -MinEt 5
New-KobeEtJobTask -Name 'Kobe_OVERNIGHT'      -Tag 'OVERNIGHT_ANALYSIS' -HourEt 21 -MinEt 0

# Weekly EOD_LEARNING on Friday 17:00 ET (converted to local)
$frLocal = Get-LocalFromET -HourEt 17 -MinuteEt 0
$friday = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday -At $frLocal
$argsWeekly = "/c cd /d `"$RepoRoot`" && $PythonExe scripts\run_job.py --tag EOD_LEARNING --dotenv $DotEnv --universe $Universe --cap $Cap --min-conf $MinConf"
if ($PerJobTelegram) { $argsWeekly += " --telegram --telegram-dotenv `"$TelegramDotenv`"" }
$actionWeekly = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument $argsWeekly
Register-ScheduledTask -TaskName 'Kobe_EOD_LEARNING' -Action $actionWeekly -Trigger $friday -Force | Out-Null
Write-Host ("Registered task: Kobe_EOD_LEARNING (Friday " + $frLocal.ToShortTimeString() + " local; ET 17:00)")

# Optional: Daily EOD_LEARNING (Mon-Fri) 17:00 ET (converted to local)
$dailyLocal = Get-LocalFromET -HourEt 17 -MinuteEt 0
New-KobeEtJobTask -Name 'Kobe_EOD_LEARNING_DAILY' -Tag 'EOD_LEARNING' -HourEt 17 -MinEt 0
Write-Host ("Registered task: Kobe_EOD_LEARNING_DAILY (" + $dailyLocal.ToShortTimeString() + " local; ET 17:00)")

# Heartbeat every minute
New-KobeHeartbeatTask

# Shadow scan at 09:45 ET
$shadowAt = Get-LocalFromET -HourEt 9 -MinuteEt 45
$shadowArgs = "/c cd /d `"$RepoRoot`" && $PythonExe scripts\run_shadow.py --dotenv $DotEnv --universe $Universe --cap $Cap"
$shadowAction = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument $shadowArgs
$shadowTrig = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $shadowAt
Register-ScheduledTask -TaskName 'Kobe_SHADOW' -Action $shadowAction -Trigger $shadowTrig -Force | Out-Null
Write-Host ("Registered task: Kobe_SHADOW (" + $shadowAt.ToShortTimeString() + " local; ET 09:45)")

# Divergence check at 10:05 ET
$divAt = Get-LocalFromET -HourEt 10 -MinuteEt 5
$divArgs = "/c cd /d `"$RepoRoot`" && $PythonExe -m monitor.divergence --telegram"
$divAction = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument $divArgs
$divTrig = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $divAt
Register-ScheduledTask -TaskName 'Kobe_DIVERGENCE' -Action $divAction -Trigger $divTrig -Force | Out-Null
Write-Host ("Registered task: Kobe_DIVERGENCE (" + $divAt.ToShortTimeString() + " local; ET 10:05)")

# Supervisor task every 10 minutes
New-KobeSupervisorTask -EveryMinutes 10

Write-Host "All tasks registered. Check Windows Task Scheduler."
