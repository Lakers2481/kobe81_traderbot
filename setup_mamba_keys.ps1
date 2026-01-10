# ╔══════════════════════════════════════════════════════════════════╗
# ║    🐍 MAMBA AI - API KEY SETUP 🐍                               ║
# ╚══════════════════════════════════════════════════════════════════╝

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
Write-Host "  🔑 SETTING UP MAMBA AI API KEYS" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
Write-Host ""

# Load .env file
$envFile = "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\.env"

if (Test-Path $envFile) {
    Write-Host "✅ Found .env file" -ForegroundColor Green
    Write-Host "   Loading API keys..." -ForegroundColor Gray
    Write-Host ""

    # Parse .env file
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        # Skip comments and empty lines
        if ($line -and -not $line.StartsWith('#')) {
            # Split on first = only
            $parts = $line.Split('=', 2)
            if ($parts.Count -eq 2) {
                $key = $parts[0].Trim()
                $value = $parts[1].Trim()

                # Set environment variable
                [System.Environment]::SetEnvironmentVariable($key, $value, [System.EnvironmentVariableTarget]::Process)
            }
        }
    }

    # Check what we loaded
    Write-Host "📋 API Keys Found:" -ForegroundColor Cyan
    Write-Host ""

    if ($env:ANTHROPIC_API_KEY) {
        $claudeKey = $env:ANTHROPIC_API_KEY
        $claudePreview = $claudeKey.Substring(0, [Math]::Min(15, $claudeKey.Length)) + "..."
        Write-Host "  ✅ CLAUDE (Anthropic)" -ForegroundColor Green
        Write-Host "     Key: $claudePreview" -ForegroundColor Gray
        Write-Host "     Status: READY TO USE" -ForegroundColor Green

        # Set as primary
        $env:AI_PROVIDER = "claude"
        $env:CLAUDE_API_KEY = $env:ANTHROPIC_API_KEY
    }

    Write-Host ""

    if ($env:OPENAI_API_KEY) {
        $openaiKey = $env:OPENAI_API_KEY
        $openaiPreview = $openaiKey.Substring(0, [Math]::Min(15, $openaiKey.Length)) + "..."
        Write-Host "  ✅ OPENAI (ChatGPT)" -ForegroundColor Green
        Write-Host "     Key: $openaiPreview" -ForegroundColor Gray
        Write-Host "     Status: READY TO USE" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  OPENAI (ChatGPT)" -ForegroundColor Yellow
        Write-Host "     Key: NOT FOUND" -ForegroundColor Gray
        Write-Host "     Status: Using Claude instead" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
    Write-Host "  ✅ SETUP COMPLETE!" -ForegroundColor Green
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🎯 MAMBA AI is now using: " -NoNewline -ForegroundColor Cyan
    Write-Host "CLAUDE (Anthropic)" -ForegroundColor Green
    Write-Host ""
    Write-Host "💬 Try it now:" -ForegroundColor Yellow
    Write-Host "   talk what is 2 plus 2?" -ForegroundColor Cyan
    Write-Host ""

} else {
    Write-Host "❌ ERROR: .env file not found!" -ForegroundColor Red
    Write-Host "   Expected at: $envFile" -ForegroundColor Red
    Write-Host ""
}
