# Quick AI Setup Checker
# Run this to see your current AI configuration status

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║           🤖 AI SETUP STATUS CHECKER 🤖                          ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check Claude API
Write-Host "┌────────────────────────────────────────────────────────────────┐" -ForegroundColor Yellow
Write-Host "│  CLAUDE (Anthropic) STATUS                                     │" -ForegroundColor Yellow
Write-Host "└────────────────────────────────────────────────────────────────┘" -ForegroundColor Yellow

if ($env:ANTHROPIC_API_KEY) {
    $claudePreview = $env:ANTHROPIC_API_KEY.Substring(0, 20) + "..."
    Write-Host "  ✅ API Key Loaded: $claudePreview" -ForegroundColor Green

    # Test Claude API
    Write-Host "  🔄 Testing Claude API connection..." -ForegroundColor Gray

    try {
        $headers = @{
            "Content-Type" = "application/json"
            "x-api-key" = $env:ANTHROPIC_API_KEY
            "anthropic-version" = "2023-06-01"
        }

        $body = @{
            model = "claude-3-opus-20240229"
            max_tokens = 50
            messages = @(
                @{
                    role = "user"
                    content = "Reply with just: OK"
                }
            )
        } | ConvertTo-Json -Depth 10

        $response = Invoke-RestMethod -Uri "https://api.anthropic.com/v1/messages" `
                                       -Method Post `
                                       -Headers $headers `
                                       -Body $body `
                                       -ErrorAction Stop

        Write-Host "  ✅ Claude API Working!" -ForegroundColor Green
        Write-Host "     Response: $($response.content[0].text)" -ForegroundColor Gray

    } catch {
        $errorMsg = $_.Exception.Message
        if ($errorMsg -like "*credit balance*") {
            Write-Host "  ❌ Claude API: NO CREDITS" -ForegroundColor Red
            Write-Host "     Add credits at: https://console.anthropic.com/settings/billing" -ForegroundColor Yellow
        } elseif ($errorMsg -like "*authentication*" -or $errorMsg -like "*invalid*") {
            Write-Host "  ❌ Claude API: INVALID KEY" -ForegroundColor Red
            Write-Host "     Get new key at: https://console.anthropic.com/" -ForegroundColor Yellow
        } else {
            Write-Host "  ❌ Claude API: ERROR" -ForegroundColor Red
            Write-Host "     Error: $errorMsg" -ForegroundColor Gray
        }
    }
} else {
    Write-Host "  ❌ No Claude API key found" -ForegroundColor Red
    Write-Host "     Add ANTHROPIC_API_KEY to .env file" -ForegroundColor Yellow
}

Write-Host ""

# Check OpenAI API
Write-Host "┌────────────────────────────────────────────────────────────────┐" -ForegroundColor Yellow
Write-Host "│  OPENAI (GPT-4) STATUS                                         │" -ForegroundColor Yellow
Write-Host "└────────────────────────────────────────────────────────────────┘" -ForegroundColor Yellow

if ($env:OPENAI_API_KEY -and $env:OPENAI_API_KEY -ne "your_openai_key_here_replace_this") {
    $openaiPreview = $env:OPENAI_API_KEY.Substring(0, 15) + "..."
    Write-Host "  ✅ API Key Loaded: $openaiPreview" -ForegroundColor Green

    # Test OpenAI API
    Write-Host "  🔄 Testing OpenAI API connection..." -ForegroundColor Gray

    try {
        $headers = @{
            "Content-Type" = "application/json"
            "Authorization" = "Bearer $env:OPENAI_API_KEY"
        }

        $body = @{
            model = "gpt-4"
            messages = @(
                @{
                    role = "user"
                    content = "Reply with just: OK"
                }
            )
            max_tokens = 10
        } | ConvertTo-Json -Depth 10

        $response = Invoke-RestMethod -Uri "https://api.openai.com/v1/chat/completions" `
                                       -Method Post `
                                       -Headers $headers `
                                       -Body $body `
                                       -ErrorAction Stop

        Write-Host "  ✅ OpenAI API Working!" -ForegroundColor Green
        Write-Host "     Response: $($response.choices[0].message.content)" -ForegroundColor Gray

    } catch {
        $errorMsg = $_.Exception.Message
        if ($errorMsg -like "*insufficient_quota*" -or $errorMsg -like "*billing*") {
            Write-Host "  ❌ OpenAI API: NO CREDITS" -ForegroundColor Red
            Write-Host "     Add credits at: https://platform.openai.com/settings/organization/billing" -ForegroundColor Yellow
        } elseif ($errorMsg -like "*authentication*" -or $errorMsg -like "*invalid*") {
            Write-Host "  ❌ OpenAI API: INVALID KEY" -ForegroundColor Red
            Write-Host "     Get new key at: https://platform.openai.com/api-keys" -ForegroundColor Yellow
        } else {
            Write-Host "  ❌ OpenAI API: ERROR" -ForegroundColor Red
            Write-Host "     Error: $errorMsg" -ForegroundColor Gray
        }
    }
} elseif ($env:OPENAI_API_KEY -eq "your_openai_key_here_replace_this") {
    Write-Host "  ⚠️  OpenAI key placeholder detected" -ForegroundColor Yellow
    Write-Host "     Replace 'your_openai_key_here_replace_this' with actual key" -ForegroundColor Yellow
    Write-Host "     Get key at: https://platform.openai.com/api-keys" -ForegroundColor Yellow
} else {
    Write-Host "  ❌ No OpenAI API key found" -ForegroundColor Red
    Write-Host "     Add OPENAI_API_KEY to .env file" -ForegroundColor Yellow
}

Write-Host ""

# Summary
Write-Host "┌────────────────────────────────────────────────────────────────┐" -ForegroundColor Cyan
Write-Host "│  SUMMARY & RECOMMENDATIONS                                     │" -ForegroundColor Cyan
Write-Host "└────────────────────────────────────────────────────────────────┘" -ForegroundColor Cyan

$claudeWorks = $false
$openaiWorks = $false

if ($env:ANTHROPIC_API_KEY) {
    try {
        $testHeaders = @{
            "Content-Type" = "application/json"
            "x-api-key" = $env:ANTHROPIC_API_KEY
            "anthropic-version" = "2023-06-01"
        }
        $testBody = @{
            model = "claude-3-opus-20240229"
            max_tokens = 10
            messages = @(@{role="user";content="test"})
        } | ConvertTo-Json -Depth 10

        $null = Invoke-RestMethod -Uri "https://api.anthropic.com/v1/messages" `
                                   -Method Post -Headers $testHeaders -Body $testBody `
                                   -ErrorAction Stop
        $claudeWorks = $true
    } catch {
        $claudeWorks = $false
    }
}

if ($env:OPENAI_API_KEY -and $env:OPENAI_API_KEY -ne "your_openai_key_here_replace_this") {
    try {
        $testHeaders = @{
            "Content-Type" = "application/json"
            "Authorization" = "Bearer $env:OPENAI_API_KEY"
        }
        $testBody = @{
            model = "gpt-4"
            messages = @(@{role="user";content="test"})
            max_tokens = 10
        } | ConvertTo-Json -Depth 10

        $null = Invoke-RestMethod -Uri "https://api.openai.com/v1/chat/completions" `
                                   -Method Post -Headers $testHeaders -Body $testBody `
                                   -ErrorAction Stop
        $openaiWorks = $true
    } catch {
        $openaiWorks = $false
    }
}

if ($claudeWorks -and $openaiWorks) {
    Write-Host "  🎉 PERFECT SETUP! Both AI providers working!" -ForegroundColor Green
    Write-Host "     Primary: Claude (Anthropic)" -ForegroundColor Gray
    Write-Host "     Fallback: OpenAI (GPT-4)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  ✅ You're ready to chat!" -ForegroundColor Green
    Write-Host "     Try: talk hello" -ForegroundColor Cyan
} elseif ($claudeWorks) {
    Write-Host "  ✅ Claude working! OpenAI not configured." -ForegroundColor Green
    Write-Host "     Primary: Claude (Anthropic)" -ForegroundColor Gray
    Write-Host "     Fallback: None" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  💡 Recommendation: Add OpenAI as fallback" -ForegroundColor Yellow
    Write-Host "     https://platform.openai.com/api-keys" -ForegroundColor Cyan
} elseif ($openaiWorks) {
    Write-Host "  ✅ OpenAI working! Claude not configured." -ForegroundColor Green
    Write-Host "     Primary: OpenAI (GPT-4)" -ForegroundColor Gray
    Write-Host "     Fallback: None" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  💡 Recommendation: Add Claude as primary" -ForegroundColor Yellow
    Write-Host "     https://console.anthropic.com/" -ForegroundColor Cyan
} else {
    Write-Host "  ❌ NO AI PROVIDERS WORKING" -ForegroundColor Red
    Write-Host ""
    Write-Host "  📋 TODO:" -ForegroundColor Yellow
    Write-Host "     1. Get OpenAI key: https://platform.openai.com/api-keys" -ForegroundColor Cyan
    Write-Host "     2. Add credits: https://platform.openai.com/settings/organization/billing" -ForegroundColor Cyan
    Write-Host "     3. Update .env: OPENAI_API_KEY=your_key_here" -ForegroundColor Cyan
    Write-Host "     4. Run: reload" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "     OR" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "     1. Add Claude credits: https://console.anthropic.com/settings/billing" -ForegroundColor Cyan
    Write-Host "     2. Run: reload" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  For complete setup guide, read:                                 ║" -ForegroundColor Cyan
Write-Host "║  AI_SETUP_COMPLETE_GUIDE.md                                      ║" -ForegroundColor Yellow
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
