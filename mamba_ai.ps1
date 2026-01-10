# ╔══════════════════════════════════════════════════════════════════╗
# ║              🐍 MAMBA AI ASSISTANT - GENIUS LEVEL 🐍              ║
# ║                   GPT-4 & Claude Powered                          ║
# ╚══════════════════════════════════════════════════════════════════╝

# Configuration
$script:OPENAI_API_KEY = $env:OPENAI_API_KEY
$script:ANTHROPIC_API_KEY = $env:ANTHROPIC_API_KEY
$script:DEFAULT_MODEL = "gpt-4"  # or "claude-3-opus-20240229"
$script:CONVERSATION_HISTORY = @()
$script:CODEBASE_PATH = "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot"

# ══════════════════════════════════════════════════════════════════
# CORE AI ENGINE
# ══════════════════════════════════════════════════════════════════

function Invoke-MambaAI {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Prompt,

        [string]$Model = $script:DEFAULT_MODEL,

        [switch]$IncludeCodebase,

        [string]$SystemPrompt = "You are Mamba AI, a genius-level coding assistant and trading expert. You help with the kobe81_traderbot trading system. Be concise, accurate, and provide working code."
    )

    Write-Host "🐍 Mamba AI thinking..." -ForegroundColor Yellow

    try {
        # Add codebase context if requested
        $fullPrompt = $Prompt
        if ($IncludeCodebase) {
            $codeContext = Get-CodebaseContext
            $fullPrompt = "CODEBASE CONTEXT:`n$codeContext`n`nUSER QUESTION: $Prompt"
        }

        # Add to conversation history
        $script:CONVERSATION_HISTORY += @{role="user"; content=$fullPrompt}

        if ($Model -like "gpt-*") {
            $response = Invoke-OpenAI -Prompt $fullPrompt -SystemPrompt $SystemPrompt
        } elseif ($Model -like "claude-*") {
            $response = Invoke-Claude -Prompt $fullPrompt -SystemPrompt $SystemPrompt
        } else {
            throw "Unknown model: $Model"
        }

        # Add AI response to history
        $script:CONVERSATION_HISTORY += @{role="assistant"; content=$response}

        Write-Host ""
        Write-Host "💡 Mamba AI:" -ForegroundColor Yellow
        Write-Host $response -ForegroundColor Cyan
        Write-Host ""

        return $response

    } catch {
        Write-Host "❌ Error: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "💡 Tip: Make sure you have OPENAI_API_KEY or ANTHROPIC_API_KEY set!" -ForegroundColor Yellow
        Write-Host "   Set it with: `$env:OPENAI_API_KEY = 'your-key-here'" -ForegroundColor Gray
        return $null
    }
}

# ══════════════════════════════════════════════════════════════════
# OPENAI GPT-4 INTEGRATION
# ══════════════════════════════════════════════════════════════════

function Invoke-OpenAI {
    param(
        [string]$Prompt,
        [string]$SystemPrompt
    )

    $headers = @{
        "Content-Type" = "application/json"
        "Authorization" = "Bearer $script:OPENAI_API_KEY"
    }

    $body = @{
        model = $script:DEFAULT_MODEL
        messages = @(
            @{role="system"; content=$SystemPrompt}
            @{role="user"; content=$Prompt}
        )
        temperature = 0.7
        max_tokens = 2000
    } | ConvertTo-Json -Depth 10

    $response = Invoke-RestMethod -Uri "https://api.openai.com/v1/chat/completions" `
                                   -Method Post `
                                   -Headers $headers `
                                   -Body $body

    return $response.choices[0].message.content
}

# ══════════════════════════════════════════════════════════════════
# CLAUDE INTEGRATION
# ══════════════════════════════════════════════════════════════════

function Invoke-Claude {
    param(
        [string]$Prompt,
        [string]$SystemPrompt
    )

    $headers = @{
        "Content-Type" = "application/json"
        "x-api-key" = $script:ANTHROPIC_API_KEY
        "anthropic-version" = "2023-06-01"
    }

    $body = @{
        model = "claude-3-opus-20240229"
        max_tokens = 2000
        system = $SystemPrompt
        messages = @(
            @{role="user"; content=$Prompt}
        )
    } | ConvertTo-Json -Depth 10

    $response = Invoke-RestMethod -Uri "https://api.anthropic.com/v1/messages" `
                                   -Method Post `
                                   -Headers $headers `
                                   -Body $body

    return $response.content[0].text
}

# ══════════════════════════════════════════════════════════════════
# CODE ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════

function Get-CodebaseContext {
    Write-Host "📂 Reading your codebase..." -ForegroundColor Gray

    $context = "KOBE81_TRADERBOT CODEBASE:`n`n"

    # Get all Python files
    $pythonFiles = Get-ChildItem -Path $script:CODEBASE_PATH -Filter "*.py" -Recurse -ErrorAction SilentlyContinue

    foreach ($file in $pythonFiles | Select-Object -First 10) {
        $relativePath = $file.FullName.Replace($script:CODEBASE_PATH, "")
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        $context += "FILE: $relativePath`n"
        $context += "```python`n$content`n```n`n"
    }

    return $context
}

# ══════════════════════════════════════════════════════════════════
# HIGH-LEVEL AI COMMANDS
# ══════════════════════════════════════════════════════════════════

function ai {
    param(
        [Parameter(ValueFromRemainingArguments=$true)]
        [string[]]$Question
    )

    $query = $Question -join " "

    if (-not $query) {
        Show-AIHelp
        return
    }

    Invoke-MambaAI -Prompt $query
}

function ai-code {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Task
    )

    $prompt = "Write production-ready code for: $Task. Include error handling and comments."
    Invoke-MambaAI -Prompt $prompt
}

function ai-fix {
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath
    )

    if (-not (Test-Path $FilePath)) {
        Write-Host "❌ File not found: $FilePath" -ForegroundColor Red
        return
    }

    $code = Get-Content $FilePath -Raw
    $prompt = "Fix any bugs or issues in this code. Explain what was wrong and provide the corrected version:`n`n\`\`\`python`n$code`n\`\`\`"

    Invoke-MambaAI -Prompt $prompt
}

function ai-review {
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath
    )

    if (-not (Test-Path $FilePath)) {
        Write-Host "❌ File not found: $FilePath" -ForegroundColor Red
        return
    }

    $code = Get-Content $FilePath -Raw
    $prompt = "Review this code like a senior developer. Check for bugs, performance issues, best practices, security. Provide specific improvements:`n`n\`\`\`python`n$code`n\`\`\`"

    Invoke-MambaAI -Prompt $prompt
}

function ai-explain {
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath
    )

    if (-not (Test-Path $FilePath)) {
        Write-Host "❌ File not found: $FilePath" -ForegroundColor Red
        return
    }

    $code = Get-Content $FilePath -Raw
    $prompt = "Explain this code in detail. What does it do? How does it work? What are the key components?`n`n\`\`\`python`n$code`n\`\`\`"

    Invoke-MambaAI -Prompt $prompt
}

function ai-strategy {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Description
    )

    $prompt = "Create a complete trading strategy for kobe81_traderbot based on: $Description. Include entry rules, exit rules, risk management, and Python implementation."
    Invoke-MambaAI -Prompt $prompt -IncludeCodebase
}

function ai-debug {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Error
    )

    $prompt = "I'm getting this error in my trading bot: $Error. Help me debug it and provide a solution."
    Invoke-MambaAI -Prompt $prompt -IncludeCodebase
}

function ai-improve {
    $prompt = "Analyze my entire kobe81_traderbot codebase and suggest 5 specific improvements to make it better."
    Invoke-MambaAI -Prompt $prompt -IncludeCodebase
}

function ai-chat {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║            🐍 MAMBA AI - INTERACTIVE CHAT MODE 🐍               ║" -ForegroundColor Yellow
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Type your questions. Type 'exit' to quit." -ForegroundColor Gray
    Write-Host ""

    while ($true) {
        Write-Host "You: " -NoNewline -ForegroundColor Green
        $input = Read-Host

        if ($input -eq "exit" -or $input -eq "quit") {
            Write-Host "👋 Goodbye!" -ForegroundColor Yellow
            break
        }

        if ($input) {
            Invoke-MambaAI -Prompt $input
        }
    }
}

function Show-AIHelp {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║              🐍 MAMBA AI - GENIUS LEVEL ASSISTANT 🐍             ║" -ForegroundColor Yellow
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🤖 AI COMMANDS:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  ai [question]              " -NoNewline; Write-Host "Ask any question" -ForegroundColor Yellow
    Write-Host "  ai-chat                    " -NoNewline; Write-Host "Interactive chat mode" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "💻 CODE COMMANDS:" -ForegroundColor Cyan
    Write-Host "  ai-code [task]             " -NoNewline; Write-Host "Write code for a task" -ForegroundColor Yellow
    Write-Host "  ai-fix [file]              " -NoNewline; Write-Host "Fix bugs in a file" -ForegroundColor Yellow
    Write-Host "  ai-review [file]           " -NoNewline; Write-Host "Review code quality" -ForegroundColor Yellow
    Write-Host "  ai-explain [file]          " -NoNewline; Write-Host "Explain what code does" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📈 TRADING COMMANDS:" -ForegroundColor Cyan
    Write-Host "  ai-strategy [description]  " -NoNewline; Write-Host "Create trading strategy" -ForegroundColor Yellow
    Write-Host "  ai-debug [error]           " -NoNewline; Write-Host "Debug trading bot errors" -ForegroundColor Yellow
    Write-Host "  ai-improve                 " -NoNewline; Write-Host "Suggest bot improvements" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  ai explain RSI indicator" -ForegroundColor Gray
    Write-Host "  ai-code a momentum scanner for stocks" -ForegroundColor Gray
    Write-Host "  ai-fix my_strategy.py" -ForegroundColor Gray
    Write-Host "  ai-review trading_bot.py" -ForegroundColor Gray
    Write-Host ""
}

# ══════════════════════════════════════════════════════════════════
# SETUP CHECK
# ══════════════════════════════════════════════════════════════════

function Test-AISetup {
    Write-Host ""
    Write-Host "🔍 Checking Mamba AI setup..." -ForegroundColor Yellow
    Write-Host ""

    if ($script:OPENAI_API_KEY) {
        Write-Host "✅ OpenAI API key found" -ForegroundColor Green
    } else {
        Write-Host "❌ OpenAI API key not set" -ForegroundColor Red
        Write-Host "   Set it with: `$env:OPENAI_API_KEY = 'your-key-here'" -ForegroundColor Gray
    }

    if ($script:ANTHROPIC_API_KEY) {
        Write-Host "✅ Claude API key found" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Claude API key not set (optional)" -ForegroundColor Yellow
    }

    if (Test-Path $script:CODEBASE_PATH) {
        $fileCount = (Get-ChildItem -Path $script:CODEBASE_PATH -Filter "*.py" -Recurse -ErrorAction SilentlyContinue).Count
        Write-Host "✅ Codebase found: $fileCount Python files" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Codebase path not found: $script:CODEBASE_PATH" -ForegroundColor Yellow
    }

    Write-Host ""
}

# ══════════════════════════════════════════════════════════════════
# AUTO-RUN ON IMPORT
# ══════════════════════════════════════════════════════════════════

Write-Host ""
Write-Host "🐍 Mamba AI loaded! Type " -NoNewline -ForegroundColor Yellow
Write-Host "ai" -NoNewline -ForegroundColor Cyan
Write-Host " for help" -ForegroundColor Yellow
Write-Host ""

# Export functions
Export-ModuleMember -Function @(
    'Invoke-MambaAI',
    'ai',
    'ai-code',
    'ai-fix',
    'ai-review',
    'ai-explain',
    'ai-strategy',
    'ai-debug',
    'ai-improve',
    'ai-chat',
    'Show-AIHelp',
    'Test-AISetup'
)
