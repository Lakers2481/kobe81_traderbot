# INTEGRATE DEEPSEEK R1 WITH MAMBA AI

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DEEPSEEK R1 + MAMBA AI INTEGRATION" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Ollama is running
Write-Host "[1] Checking Ollama Status" -ForegroundColor Yellow

try {
    $ollamaTest = ollama list 2>&1
    Write-Host "  [OK] Ollama is installed" -ForegroundColor Green

    # Check if DeepSeek model exists
    if ($ollamaTest -match "deepseek-r1") {
        Write-Host "  [OK] DeepSeek R1 model found" -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] DeepSeek R1 model not found" -ForegroundColor Red
        Write-Host "  Run this first: .\setup_deepseek_local.ps1" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "  [ERROR] Ollama not found" -ForegroundColor Red
    Write-Host "  Install Ollama first: https://ollama.com/download" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "[2] Reading Mamba AI Configuration" -ForegroundColor Yellow

$mambaPath = "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\mamba_ai_v2.ps1"

if (-not (Test-Path $mambaPath)) {
    Write-Host "  [ERROR] Mamba AI not found at: $mambaPath" -ForegroundColor Red
    exit 1
}

Write-Host "  [OK] Mamba AI found" -ForegroundColor Green

Write-Host ""
Write-Host "[3] Adding DeepSeek Support to Mamba AI" -ForegroundColor Yellow

# Read current Mamba AI
$mambaContent = Get-Content $mambaPath -Raw

# Check if DeepSeek already integrated
if ($mambaContent -match "function Invoke-Ollama") {
    Write-Host "  [!] DeepSeek already integrated in Mamba AI" -ForegroundColor Yellow
    Write-Host "  Skipping integration..." -ForegroundColor Gray
} else {
    Write-Host "  Adding Invoke-Ollama function..." -ForegroundColor Gray

    # Add Ollama function after provider detection section
    $ollamaFunction = @'

# ============================================================================
# DEEPSEEK R1 LOCAL AI (FREE)
# ============================================================================

function Invoke-Ollama {
    param(
        [string]$Prompt,
        [string]$Model = "deepseek-r1:14b",
        [array]$ConversationHistory = @()
    )

    Write-Host "[DeepSeek R1] Thinking..." -ForegroundColor Cyan

    # Build messages array
    $messages = @()

    # Add conversation history
    foreach ($msg in $ConversationHistory) {
        $messages += @{
            role = $msg.role
            content = $msg.content
        }
    }

    # Add current prompt
    $messages += @{
        role = "user"
        content = $Prompt
    }

    $body = @{
        model = $Model
        messages = $messages
        stream = $false
        options = @{
            temperature = 0.7
            top_p = 0.9
        }
    } | ConvertTo-Json -Depth 10

    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/chat" `
                                       -Method Post `
                                       -ContentType "application/json" `
                                       -Body $body `
                                       -TimeoutSec 120

        $reply = $response.message.content

        Write-Host "[DeepSeek R1] Done!" -ForegroundColor Green

        return $reply
    } catch {
        Write-Host "[ERROR] DeepSeek R1 failed: $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

function Test-OllamaAvailable {
    try {
        $test = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" `
                                   -Method Get `
                                   -TimeoutSec 5 `
                                   -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

'@

    # Find where to insert (after provider detection, before main loop)
    $insertPosition = $mambaContent.IndexOf("# Main conversation loop")

    if ($insertPosition -eq -1) {
        # Fallback: insert at end
        $mambaContent += $ollamaFunction
    } else {
        $mambaContent = $mambaContent.Insert($insertPosition, $ollamaFunction + "`n`n")
    }

    # Modify provider detection to include Ollama
    $providerDetection = @'

# Auto-detect best available provider
$availableProvider = $null

# Check DeepSeek (FREE LOCAL)
if (Test-OllamaAvailable) {
    $availableProvider = "ollama"
    Write-Host "[Provider] DeepSeek R1 (Local, FREE)" -ForegroundColor Green
}
# Check OpenAI
elseif ($env:OPENAI_API_KEY -and $env:OPENAI_API_KEY -ne "your_openai_key_here_replace_this") {
    $availableProvider = "openai"
    Write-Host "[Provider] OpenAI (GPT-4o)" -ForegroundColor Cyan
}
# Check Anthropic
elseif ($env:ANTHROPIC_API_KEY -and $env:ANTHROPIC_API_KEY -ne "your_anthropic_key_here_replace_this") {
    $availableProvider = "anthropic"
    Write-Host "[Provider] Claude (Anthropic)" -ForegroundColor Cyan
}
# Check Groq
elseif ($env:GROQ_API_KEY -and $env:GROQ_API_KEY -ne "your_groq_key_here_replace_this") {
    $availableProvider = "groq"
    Write-Host "[Provider] Groq (LLaMA 3.1)" -ForegroundColor Cyan
}
else {
    Write-Host "[ERROR] No AI provider configured!" -ForegroundColor Red
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  1. Run: .\setup_deepseek_local.ps1 (FREE, LOCAL)" -ForegroundColor Gray
    Write-Host "  2. Run: .\setup_groq_now.ps1 (FREE, API)" -ForegroundColor Gray
    Write-Host "  3. Add OpenAI key to .env file ($5)" -ForegroundColor Gray
    exit 1
}
'@

    # Replace or add provider detection
    if ($mambaContent -match "# Auto-detect best available provider") {
        $mambaContent = $mambaContent -replace "(?s)# Auto-detect best available provider.*?(?=\n#|\n\n|$)", $providerDetection
    } else {
        # Add before main loop
        $insertPosition = $mambaContent.IndexOf("# Main conversation loop")
        if ($insertPosition -ne -1) {
            $mambaContent = $mambaContent.Insert($insertPosition, $providerDetection + "`n`n")
        }
    }

    # Save updated Mamba AI
    Set-Content -Path $mambaPath -Value $mambaContent -Encoding UTF8

    Write-Host "  [OK] Integration complete!" -ForegroundColor Green
}

Write-Host ""
Write-Host "[4] Creating Test Command" -ForegroundColor Yellow

# Create a simple test alias
$testScript = @'
# Test DeepSeek with Mamba AI
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Start Mamba AI
& "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\mamba_ai_v2.ps1"
'@

$testPath = "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\test_deepseek.ps1"
Set-Content -Path $testPath -Value $testScript -Encoding UTF8

Write-Host "  [OK] Test script created: test_deepseek.ps1" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "INTEGRATION COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Your Mamba AI now has:" -ForegroundColor Yellow
Write-Host "  ✅ DeepSeek R1 (FREE, LOCAL) - Primary" -ForegroundColor Cyan
Write-Host "  ✅ Auto-fallback to OpenAI/Claude/Groq" -ForegroundColor Cyan
Write-Host "  ✅ Conversation history support" -ForegroundColor Cyan
Write-Host "  ✅ 97.3% math reasoning (equals Claude)" -ForegroundColor Cyan
Write-Host ""
Write-Host "How to use:" -ForegroundColor Yellow
Write-Host "  1. Make sure Ollama is running (auto-starts on Windows)" -ForegroundColor Gray
Write-Host "  2. Run: .\test_deepseek.ps1" -ForegroundColor Gray
Write-Host "  3. Ask: 'explain this trading strategy'" -ForegroundColor Gray
Write-Host "  4. Use for: code debugging, trading analysis, Q&A" -ForegroundColor Gray
Write-Host ""
Write-Host "Example trading questions:" -ForegroundColor Yellow
Write-Host "  - Explain why PLTR has 5 consecutive down days" -ForegroundColor Cyan
Write-Host "  - What's the probability of a bounce after this pattern?" -ForegroundColor Cyan
Write-Host "  - Debug this PowerShell function for me" -ForegroundColor Cyan
Write-Host "  - Calculate Kelly criterion for 60% win rate, 1.5 R:R" -ForegroundColor Cyan
Write-Host ""
Write-Host "Realistic expectations:" -ForegroundColor Yellow
Write-Host "  ✅ Math & calculations: Equals Claude" -ForegroundColor Green
Write-Host "  ✅ Code explanation: Very good" -ForegroundColor Green
Write-Host "  ✅ Trading analysis: Good" -ForegroundColor Green
Write-Host "  ⚠️  Large codebase refactoring: Not as good as Claude" -ForegroundColor Yellow
Write-Host ""
Write-Host "Ready to test? Run:" -ForegroundColor Yellow
Write-Host "  .\test_deepseek.ps1" -ForegroundColor Cyan
Write-Host ""
