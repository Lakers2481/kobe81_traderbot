# Search for all API keys in the system

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "           API KEY FINDER                                     " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Check environment variables
Write-Host "----------------------------------------------------------------" -ForegroundColor Yellow
Write-Host "  ENVIRONMENT VARIABLES (Currently Loaded)                      " -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------" -ForegroundColor Yellow

$apiKeys = @{
    "OPENAI_API_KEY" = $env:OPENAI_API_KEY
    "CODEX_API_KEY" = $env:CODEX_API_KEY
    "ANTHROPIC_API_KEY" = $env:ANTHROPIC_API_KEY
    "OPENAI_KEY" = $env:OPENAI_KEY
    "AZURE_OPENAI_KEY" = $env:AZURE_OPENAI_KEY
    "AZURE_OPENAI_API_KEY" = $env:AZURE_OPENAI_API_KEY
    "HUGGINGFACE_TOKEN" = $env:HUGGINGFACE_TOKEN
    "HF_TOKEN" = $env:HF_TOKEN
    "GEMINI_API_KEY" = $env:GEMINI_API_KEY
    "GOOGLE_API_KEY" = $env:GOOGLE_API_KEY
    "COHERE_API_KEY" = $env:COHERE_API_KEY
    "AI21_API_KEY" = $env:AI21_API_KEY
    "MISTRAL_API_KEY" = $env:MISTRAL_API_KEY
}

$foundKeys = @()

foreach ($key in $apiKeys.Keys) {
    if ($apiKeys[$key]) {
        if ($apiKeys[$key] -eq "your_openai_key_here_replace_this" -or
            $apiKeys[$key] -like "*placeholder*" -or
            $apiKeys[$key] -like "*replace*") {
            Write-Host "  ⚠️  $key" -NoNewline -ForegroundColor Yellow
            Write-Host " = PLACEHOLDER" -ForegroundColor Gray
        } else {
            $preview = $apiKeys[$key].Substring(0, [Math]::Min(20, $apiKeys[$key].Length)) + "..."
            Write-Host "  ✅ $key" -NoNewline -ForegroundColor Green
            Write-Host " = $preview" -ForegroundColor Gray
            $foundKeys += $key
        }
    } else {
        Write-Host "  ❌ $key" -NoNewline -ForegroundColor Red
        Write-Host " = NOT SET" -ForegroundColor Gray
    }
}

Write-Host ""

# Check .env file
Write-Host "----------------------------------------------------------------" -ForegroundColor Yellow
Write-Host "  .ENV FILE (On Disk)                                           " -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------" -ForegroundColor Yellow

$envPath = "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\.env"
if (Test-Path $envPath) {
    $envContent = Get-Content $envPath

    $envKeys = @(
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_KEY",
        "AZURE_OPENAI_KEY",
        "AZURE_OPENAI_API_KEY",
        "HUGGINGFACE_TOKEN",
        "HF_TOKEN",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "COHERE_API_KEY",
        "AI21_API_KEY",
        "MISTRAL_API_KEY"
    )

    foreach ($keyName in $envKeys) {
        $line = $envContent | Where-Object { $_ -match "^$keyName\s*=" }
        if ($line) {
            $value = ($line -split '=', 2)[1].Trim()
            if ($value -and $value -ne "" -and
                $value -notlike "*your_*_here*" -and
                $value -notlike "*replace*" -and
                $value -notlike "*placeholder*") {

                $preview = $value.Substring(0, [Math]::Min(20, $value.Length)) + "..."
                Write-Host "  ✅ $keyName" -NoNewline -ForegroundColor Green
                Write-Host " = $preview" -ForegroundColor Gray
            } elseif ($value -like "*your_*_here*" -or $value -like "*replace*" -or $value -like "*placeholder*") {
                Write-Host "  ⚠️  $keyName" -NoNewline -ForegroundColor Yellow
                Write-Host " = PLACEHOLDER" -ForegroundColor Gray
            } else {
                Write-Host "  ⚠️  $keyName" -NoNewline -ForegroundColor Yellow
                Write-Host " = EMPTY" -ForegroundColor Gray
            }
        }
    }
} else {
    Write-Host "  ❌ .env file not found!" -ForegroundColor Red
}

Write-Host ""

# Search for other .env files
Write-Host "----------------------------------------------------------------" -ForegroundColor Yellow
Write-Host "  OTHER CONFIG FILES                                            " -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------" -ForegroundColor Yellow

$searchPaths = @(
    "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot",
    "C:\Users\Owner\OneDrive\Desktop",
    "C:\Users\Owner",
    "C:\Users\Owner\Documents"
)

$foundFiles = @()

foreach ($path in $searchPaths) {
    if (Test-Path $path) {
        # Search for .env files (but not .env.template)
        $envFiles = Get-ChildItem -Path $path -Filter ".env*" -File -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -ne ".env.template" -and $_.Name -ne ".env.example" }

        foreach ($file in $envFiles) {
            Write-Host "  📄 Found: $($file.FullName)" -ForegroundColor Cyan
            $foundFiles += $file.FullName
        }

        # Search for credentials.json, secrets.json, etc.
        $credFiles = Get-ChildItem -Path $path -Filter "*credential*" -File -ErrorAction SilentlyContinue
        foreach ($file in $credFiles) {
            Write-Host "  📄 Found: $($file.FullName)" -ForegroundColor Cyan
            $foundFiles += $file.FullName
        }
    }
}

if ($foundFiles.Count -eq 0) {
    Write-Host "  ℹ️  No additional config files found" -ForegroundColor Gray
}

Write-Host ""

# Summary
Write-Host "----------------------------------------------------------------" -ForegroundColor Cyan
Write-Host "  SUMMARY                                                       " -ForegroundColor Cyan
Write-Host "----------------------------------------------------------------" -ForegroundColor Cyan

if ($foundKeys.Count -gt 0) {
    Write-Host ""
    Write-Host "  ✅ FOUND $($foundKeys.Count) VALID API KEY(S):" -ForegroundColor Green
    foreach ($key in $foundKeys) {
        Write-Host "     - $key" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "  💡 These keys are already loaded and ready to use!" -ForegroundColor Cyan
    Write-Host "     Just run: reload" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "  ❌ NO VALID API KEYS FOUND" -ForegroundColor Red
    Write-Host ""
    Write-Host "  📋 You need to:" -ForegroundColor Yellow
    Write-Host "     1. Get an API key from one of these providers:" -ForegroundColor Gray
    Write-Host "        • OpenAI: https://platform.openai.com/api-keys" -ForegroundColor Cyan
    Write-Host "        • Anthropic (Claude): https://console.anthropic.com/" -ForegroundColor Cyan
    Write-Host "        • Azure OpenAI: https://portal.azure.com/" -ForegroundColor Cyan
    Write-Host "        • Google AI: https://makersuite.google.com/app/apikey" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "     2. Add it to your .env file:" -ForegroundColor Gray
    Write-Host "        OPENAI_API_KEY=your_key_here" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "     3. Reload PowerShell:" -ForegroundColor Gray
    Write-Host "        reload" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  For detailed setup, read: AI_SETUP_COMPLETE_GUIDE.md" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
