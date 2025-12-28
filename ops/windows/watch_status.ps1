param(
  [string]$RepoRoot = "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot",
  [string]$PythonExe = "python"
)

Write-Host "Starting Kobe file watcher at $RepoRoot"

$fsw = New-Object IO.FileSystemWatcher $RepoRoot
$fsw.IncludeSubdirectories = $true
$fsw.Filter = '*.*'

$debounce = New-Object Collections.Concurrent.ConcurrentDictionary[string, datetime]

function Debounced-Run($actionKey, [ScriptBlock]$action, $ms=2000) {
  $now = Get-Date
  $debounce[$actionKey] = $now
  Start-Sleep -Milliseconds $ms
  $latest = $debounce[$actionKey]
  if ($latest -eq $now) {
    & $action
  }
}

$handler = Register-ObjectEvent $fsw Changed -Action {
  param($sender, $eventArgs)
  $path = $eventArgs.FullPath
  # Filter noisy/irrelevant changes
  if ($path -match '\\.git\\' -or $path -match '\\__pycache__\\' -or $path -match '\\.pytest_cache\\') { return }
  if (-not ($path -match '\\.py$' -or $path -match '\\.md$' -or $path -match '\\.y(a)?ml$' -or $path -match '\\.toml$')) { return }

  Debounced-Run 'status_update' {
    try {
      & $using:PythonExe "$using:RepoRoot\scripts\log_event.py" --event code_changed --payload "{\"path\": \"$path\"}"
    } catch {}
    try {
      & $using:PythonExe "$using:RepoRoot\scripts\update_status_md.py"
    } catch {}
    Write-Host "STATUS.md updated due to change: $path"
  } 5000
}

$handler2 = Register-ObjectEvent $fsw Created -Action {
  param($sender, $eventArgs)
  $path = $eventArgs.FullPath
  if ($path -match '\\.git\\' -or $path -match '\\__pycache__\\' -or $path -match '\\.pytest_cache\\') { return }
  if (-not ($path -match '\\.py$' -or $path -match '\\.md$' -or $path -match '\\.y(a)?ml$' -or $path -match '\\.toml$')) { return }
  Debounced-Run 'status_update_created' {
    try { & $using:PythonExe "$using:RepoRoot\scripts\log_event.py" --event code_added --payload "{\"path\": \"$path\"}" } catch {}
    try { & $using:PythonExe "$using:RepoRoot\scripts\update_status_md.py" } catch {}
    Write-Host "STATUS.md updated after new file: $path"
  } 5000
}

while ($true) { Start-Sleep -Seconds 5 }

