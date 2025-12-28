param(
  [string]$RepoRoot = "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot",
  [string]$PythonExe = "python",
  [string]$DotEnv = ".\\.env",
  [int]$RefreshSeconds = 5,
  [int]$ShowEvents = 8,
  [int]$ShowDecisions = 4,
  [string]$TimeZoneId = 'Central Standard Time'
)

$ErrorActionPreference = 'SilentlyContinue'

function Tail-File([string]$Path, [int]$Lines = 10) {
  if (Test-Path $Path) { Get-Content $Path -Tail $Lines } else { @() }
}

function Read-JsonLast([string]$Path) {
  if (-not (Test-Path $Path)) { return $null }
  try {
    $line = (Get-Content $Path -Tail 1)
    if (-not $line) { return $null }
    return $line | ConvertFrom-Json
  } catch { return $null }
}

Set-Location $RepoRoot | Out-Null

# Previous-state trackers for alerts
$script:lastKs = $null
$script:lastTotdMtime = $null
$script:lastPicksMtime = $null

function Beep-Alert([int]$freq = 1200, [int]$dur = 200) {
  try { [console]::beep($freq, $dur) } catch {}
}

function Get-TaskRow($rows, [string]$name) {
  return ($rows | Where-Object { $_.Name -eq $name } | Select-Object -First 1)
}

function Check-Deadline([datetime]$nowLocal, [pscustomobject]$row, [datetime]$deadline, [string]$label, [switch]$warn, [switch]$alert) {
  if (-not $row) { return }
  if (($nowLocal -ge $deadline) -and ($row.LastRun.Date -ne $nowLocal.Date)) {
    if ($alert) {
      Write-Host ("  [ALERT] " + $label + " not executed today (deadline " + $deadline.ToShortTimeString() + ")") -ForegroundColor Red
      Beep-Alert 1200 250
    } elseif ($warn) {
      Write-Host ("  [WARN] " + $label + " not executed yet (deadline " + $deadline.ToShortTimeString() + ")") -ForegroundColor Yellow
      Beep-Alert 1000 150
    }
  }
}

try { $tz = [System.TimeZoneInfo]::FindSystemTimeZoneById($TimeZoneId) } catch { $tz = $null }

while ($true) {
  try {
    # Invoke heartbeat to refresh snapshot
    & $PythonExe "scripts/heartbeat.py" --dotenv $DotEnv | Out-Null
  } catch {}

  Clear-Host
  $nowCT = if ($tz) { [System.TimeZoneInfo]::ConvertTime([DateTime]::UtcNow, $tz) } else { Get-Date }
  Write-Host ("KOBE LIVE DASHBOARD  â€”  " + $nowCT.ToString('yyyy-MM-dd h:mm:ss tt') + " (CT)") -ForegroundColor Cyan
  try { $tzET = [System.TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time') } catch { $tzET = $null }
  $nowET = if ($tzET) { [System.TimeZoneInfo]::ConvertTime([DateTime]::UtcNow, $tzET) } else { $nowCT.AddHours(1) }
  Write-Host ("ET: " + $nowET.ToString('h:mm tt') + " (ET)")
  Write-Host ("Repo: " + $RepoRoot)

  $hbTxt = Join-Path $RepoRoot "logs/heartbeat_latest.txt"
  $hbJson = Join-Path $RepoRoot "logs/heartbeat.jsonl"
  $evtLog = Join-Path $RepoRoot "logs/events.jsonl"
  $decLog = Join-Path $RepoRoot "logs/decisions.jsonl"
  $picksCsv = Join-Path $RepoRoot "logs/daily_picks.csv"
  $totdCsv  = Join-Path $RepoRoot "logs/trade_of_day.csv"

  $hb = Read-JsonLast $hbJson
  Write-Host ""; Write-Host "HEARTBEAT" -ForegroundColor Yellow
  if ($hb -ne $null) {
    $etStr = $hb.ts_et
    Write-Host ("  ET: " + $etStr + " | CT: " + $nowCT.ToString('yyyy-MM-dd h:mm:ss tt'))
  } else {
    $line = if (Test-Path $hbTxt) { Get-Content $hbTxt -Tail 1 } else { "(no heartbeat yet)" }
    Write-Host ("  " + $line)
  }

  if ($hb -ne $null) {
    $ks = if ($hb.kill_switch) { 'ON' } else { 'off' }
    $picks = if ($hb.files.daily_picks.exists) { 'Y' } else { 'n' }
    $totd = if ($hb.files.totd.exists) { 'Y' } else { 'n' }
    $mrep = if ($hb.files.morning_report.exists) { 'Y' } else { 'n' }
    $mchk = if ($hb.files.morning_check.exists) { 'Y' } else { 'n' }
    $freeGB = if ($hb.disk.free) { [math]::Round(($hb.disk.free/1GB),1) } else { $null }
    Write-Host ("  KS=" + $ks + " | picks=" + $picks + " | totd=" + $totd + " | mrep=" + $mrep + " | mchk=" + $mchk + " | free=" + $freeGB + " GB")

    # Kill Switch alert on change
    if ($null -ne $script:lastKs) {
      if (($hb.kill_switch) -and (-not $script:lastKs)) {
        Write-Host "  [ALERT] Kill Switch is ON" -ForegroundColor Red
        Beep-Alert 1400 250
      } elseif ((-not $hb.kill_switch) -and $script:lastKs) {
        Write-Host "  Kill Switch turned OFF" -ForegroundColor Green
        Beep-Alert 900 200
      }
    }
    $script:lastKs = [bool]$hb.kill_switch
  }

  Write-Host ""; Write-Host "SCHEDULE (Kobe_ tasks)" -ForegroundColor Yellow
  try {
    $tasks = Get-ScheduledTask | Where-Object { $_.TaskName -like 'Kobe_*' } | Sort-Object TaskName
    $rows = @()
    foreach ($t in $tasks) {
      $info = Get-ScheduledTaskInfo -TaskName $t.TaskName
      $rows += [PSCustomObject]@{
        Name = $t.TaskName
        LastRun = $info.LastRunTime
        NextRun = $info.NextRunTime
        State = $t.State
        LastResult = $info.LastTaskResult
      }
    }
    $rows | Select-Object Name,LastRun,NextRun,State,LastResult | Format-Table -AutoSize | Out-String | Write-Host

    # Alerts for failed tasks (LastTaskResult != 0)
    $failed = $rows | Where-Object { $_.LastResult -ne 0 }
    if ($failed.Count -gt 0) {
      Write-Host "  [ALERT] Failed tasks detected:" -ForegroundColor Red
      foreach ($f in $failed) { Write-Host ("   - " + $f.Name + " (LastResult=" + $f.LastResult + ")") -ForegroundColor Red }
      Beep-Alert 800 250
    }

    # Time-based checks (local time)
    $nowLocal = $nowCT
    # Morning report missing after 06:50
    if ($nowLocal.Hour -gt 6 -or ($nowLocal.Hour -eq 6 -and $nowLocal.Minute -ge 50)) {
      if (-not (Test-Path $mrep)) {
        Write-Host "  [WARN] Morning report not generated yet (after 06:50)" -ForegroundColor Yellow
        Beep-Alert 1000 140
      }
    }
    # Morning check missing after 07:05
    if ($nowLocal.Hour -gt 7 -or ($nowLocal.Hour -eq 7 -and $nowLocal.Minute -ge 5)) {
      if (-not (Test-Path $mchk)) {
        Write-Host "  [WARN] Morning check not written yet (after 07:05)" -ForegroundColor Yellow
        Beep-Alert 950 140
      }
    }
    # Deadlines for key tasks (local time)
    $today = Get-Date -Date $nowCT.Date
    Check-Deadline $nowLocal (Get-TaskRow $rows 'Kobe_DB_BACKUP')      ($today.AddHours(5).AddMinutes(45))  'DB_BACKUP' -warn
    Check-Deadline $nowLocal (Get-TaskRow $rows 'Kobe_DATA_UPDATE')    ($today.AddHours(6).AddMinutes(30))  'DATA_UPDATE' -warn
    Check-Deadline $nowLocal (Get-TaskRow $rows 'Kobe_PRE_GAME')       ($today.AddHours(8).AddMinutes(15))  'PRE_GAME' -warn
    Check-Deadline $nowLocal (Get-TaskRow $rows 'Kobe_MARKET_NEWS')    ($today.AddHours(9).AddMinutes(10))  'MARKET_NEWS' -warn
    Check-Deadline $nowLocal (Get-TaskRow $rows 'Kobe_PREMARKET_SCAN') ($today.AddHours(9).AddMinutes(25))  'PREMARKET_SCAN' -warn
    Check-Deadline $nowLocal (Get-TaskRow $rows 'Kobe_FIRST_SCAN')     ($today.AddHours(10).AddMinutes(10)) 'FIRST_SCAN' -alert
    Check-Deadline $nowLocal (Get-TaskRow $rows 'Kobe_HALF_TIME')      ($today.AddHours(12).AddMinutes(15)) 'HALF_TIME' -warn
    Check-Deadline $nowLocal (Get-TaskRow $rows 'Kobe_AFTERNOON_SCAN') ($today.AddHours(14).AddMinutes(45)) 'AFTERNOON_SCAN' -warn
    Check-Deadline $nowLocal (Get-TaskRow $rows 'Kobe_SWING_SCANNER')  ($today.AddHours(15).AddMinutes(40)) 'SWING_SCANNER' -warn
    Check-Deadline $nowLocal (Get-TaskRow $rows 'Kobe_POST_GAME')      ($today.AddHours(16).AddMinutes(10)) 'POST_GAME' -warn
    # EOD_REPORT must run by 16:15
    Check-Deadline $nowLocal (Get-TaskRow $rows 'Kobe_EOD_REPORT')     ($today.AddHours(16).AddMinutes(15)) 'EOD_REPORT' -alert
    # EOD_LEARNING Fridays by 17:15
    if ($nowLocal.DayOfWeek -eq 'Friday') {
      Check-Deadline $nowLocal (Get-TaskRow $rows 'Kobe_EOD_LEARNING') ($today.AddHours(17).AddMinutes(15)) 'EOD_LEARNING' -warn
    }
    # OVERNIGHT_ANALYSIS by 21:30
    Check-Deadline $nowLocal (Get-TaskRow $rows 'Kobe_OVERNIGHT')      ($today.AddHours(21).AddMinutes(30)) 'OVERNIGHT_ANALYSIS' -warn
  } catch {}

  Write-Host ""; Write-Host ("EVENTS (last " + $ShowEvents + ")") -ForegroundColor Yellow
  $ev = Tail-File $evtLog $ShowEvents
  foreach ($e in $ev) { Write-Host ("  " + $e) }

  Write-Host ""; Write-Host ("DECISIONS (last " + $ShowDecisions + ")") -ForegroundColor Yellow
  $dc = Tail-File $decLog $ShowDecisions
  foreach ($d in $dc) { Write-Host ("  " + $d) }

  # Show Top-3 picks and TOTD if available
  Write-Host ""
  if (Test-Path $picksCsv) {
    Write-Host "TOP-3 PICKS" -ForegroundColor Yellow
    try {
      $pItem = Get-Item $picksCsv
      $p = Import-Csv $picksCsv
      # Alert on update (mtime change)
      if ($script:lastPicksMtime -ne $pItem.LastWriteTime) {
        if ($script:lastPicksMtime -ne $null) {
          Write-Host ("  [UPDATE] daily_picks.csv refreshed: " + $pItem.LastWriteTime) -ForegroundColor Green
          Beep-Alert 1100 180
        }
        $script:lastPicksMtime = $pItem.LastWriteTime
      }
      if ($p) {
        # Choose common columns if present
        $cols = @('strategy','symbol','side','entry_price','stop_loss','take_profit','conf_score') | Where-Object { $p[0].psobject.Properties.Name -contains $_ }
        if ($cols.Count -eq 0) { $cols = $p[0].psobject.Properties.Name }
        $p | Select-Object -First 3 $cols | Format-Table -AutoSize | Out-String | Write-Host
      }
    } catch { Write-Host "  (error reading daily_picks.csv)" }
  } else {
    Write-Host "TOP-3 PICKS" -ForegroundColor Yellow
    Write-Host "  (no daily_picks.csv yet)"
  }

  Write-Host ""
  if (Test-Path $totdCsv) {
    Write-Host "TRADE OF THE DAY" -ForegroundColor Yellow
    try {
      $tItem = Get-Item $totdCsv
      $t = Import-Csv $totdCsv
      # Alert on update (mtime change)
      if ($script:lastTotdMtime -ne $tItem.LastWriteTime) {
        if ($script:lastTotdMtime -ne $null) {
          Write-Host ("  [UPDATE] trade_of_day.csv refreshed: " + $tItem.LastWriteTime) -ForegroundColor Green
          # Double beep for TOTD
          Beep-Alert 1300 150; Start-Sleep -Milliseconds 80; Beep-Alert 1500 150
        }
        $script:lastTotdMtime = $tItem.LastWriteTime
      }
      if ($t) {
        $cols = @('strategy','symbol','side','entry_price','stop_loss','take_profit','conf_score') | Where-Object { $t[0].psobject.Properties.Name -contains $_ }
        if ($cols.Count -eq 0) { $cols = $t[0].psobject.Properties.Name }
        $t | Select-Object -First 1 $cols | Format-Table -AutoSize | Out-String | Write-Host
      }
    } catch { Write-Host "  (error reading trade_of_day.csv)" }
  } else {
    Write-Host "TRADE OF THE DAY" -ForegroundColor Yellow
    Write-Host "  (no trade_of_day.csv yet)"
  }

  Write-Host ""; Write-Host ("Refresh in " + $RefreshSeconds + "s ...  (Ctrl+C to stop)") -ForegroundColor DarkGray
  Start-Sleep -Seconds $RefreshSeconds
}
