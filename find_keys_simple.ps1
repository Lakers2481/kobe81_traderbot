# Simple API Key Finder

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "API KEY FINDER" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Checking Environment Variables..." -ForegroundColor Yellow
Write-Host ""

$keys = @{
    "ANTHROPIC_API_KEY" = $env:ANTHROPIC_API_KEY
    "OPENAI_API_KEY" = $env:OPENAI_API_KEY
    "CODEX_API_KEY" = $env:CODEX_API_KEY
    "AZURE_OPENAI_KEY" = $env:AZURE_OPENAI_KEY
    "GOOGLE_API_KEY" = $env:GOOGLE_API_KEY
    "GEMINI_API_KEY" = $env:GEMINI_API_KEY
    "HF_TOKEN" = $env:HF_TOKEN
    "COHERE_API_KEY" = $env:COHERE_API_KEY
}

$foundInEnv = @()

foreach ($name in $keys.Keys) {
    $value = $keys[$name]
    if ($value -and $value -ne "" -and $value -ne "your_openai_key_here_replace_this") {
        $preview = $value.Substring(0, [Math]::Min(15, $value.Length)) + "..."
        Write-Host "  [OK] $name = $preview" -ForegroundColor Green
        $foundInEnv += $name
    } else {
        Write-Host "  [--] $name = NOT SET" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "Checking .env file..." -ForegroundColor Yellow
Write-Host ""

$envPath = "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\.env"
if (Test-Path $envPath) {
    $content = Get-Content $envPath

    $foundInFile = @()

    foreach ($line in $content) {
        if ($line -match "^(ANTHROPIC_API_KEY|OPENAI_API_KEY|CODEX_API_KEY|AZURE_OPENAI_KEY|GOOGLE_API_KEY|GEMINI_API_KEY|HF_TOKEN|COHERE_API_KEY)\s*=\s*(.+)$") {
            $keyName = $matches[1]
            $keyValue = $matches[2].Trim()

            if ($keyValue -ne "" -and $keyValue -ne "your_openai_key_here_replace_this") {
                $preview = $keyValue.Substring(0, [Math]::Min(15, $keyValue.Length)) + "..."
                Write-Host "  [OK] $keyName = $preview" -ForegroundColor Green
                $foundInFile += $keyName
            }
        }
    }

    if ($foundInFile.Count -eq 0) {
        Write-Host "  No valid AI API keys found in .env" -ForegroundColor Gray
    }
} else {
    Write-Host "  .env file not found!" -ForegroundColor Red
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

if ($foundInEnv.Count -gt 0) {
    Write-Host "[SUCCESS] Found $($foundInEnv.Count) API key(s) loaded:" -ForegroundColor Green
    foreach ($key in $foundInEnv) {
        Write-Host "  - $key" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "These keys are ready to use!" -ForegroundColor Cyan
} else {
    Write-Host "[WARNING] No AI API keys found in environment" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To add a key:" -ForegroundColor Gray
    Write-Host "  1. Get an API key from a provider" -ForegroundColor Gray
    Write-Host "  2. Add to .env file: OPENAI_API_KEY=your_key" -ForegroundColor Gray
    Write-Host "  3. Run: reload" -ForegroundColor Gray
}

Write-Host ""
