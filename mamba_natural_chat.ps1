# ╔══════════════════════════════════════════════════════════════════╗
# ║    🐍 MAMBA AI - ULTRA-SIMPLE CHAT MODE 🐍                      ║
# ╚══════════════════════════════════════════════════════════════════╝

function Start-Chat {
    <#
    .SYNOPSIS
    Start interactive chat session with Mamba AI
    #>

    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║           💬 MAMBA AI CHAT SESSION STARTED 💬                   ║" -ForegroundColor Cyan
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🤖 Using: " -NoNewline -ForegroundColor Gray
    Write-Host "Claude (Anthropic)" -ForegroundColor Green
    Write-Host ""
    Write-Host "💡 Tips:" -ForegroundColor Yellow
    Write-Host "   • Just type naturally - no special syntax!" -ForegroundColor Gray
    Write-Host "   • Type 'exit' or 'quit' to end chat" -ForegroundColor Gray
    Write-Host "   • Type 'clear' to clear screen" -ForegroundColor Gray
    Write-Host "   • Ask about your trading bot!" -ForegroundColor Gray
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor DarkGray
    Write-Host ""

    $chatting = $true
    while ($chatting) {
        # Prompt
        Write-Host "You: " -NoNewline -ForegroundColor Yellow
        $input = Read-Host

        # Handle commands
        if ($input -eq 'exit' -or $input -eq 'quit') {
            Write-Host ""
            Write-Host "👋 Chat session ended. Type " -NoNewline -ForegroundColor Gray
            Write-Host "chat" -NoNewline -ForegroundColor Cyan
            Write-Host " to start again!" -ForegroundColor Gray
            Write-Host ""
            $chatting = $false
            continue
        }

        if ($input -eq 'clear') {
            Clear-Host
            Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
            Write-Host "║           💬 MAMBA AI CHAT SESSION 💬                           ║" -ForegroundColor Cyan
            Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
            Write-Host ""
            continue
        }

        if (-not $input) {
            continue
        }

        # Call AI
        Write-Host ""
        ai $input
    }
}

# Aliases
Set-Alias -Name chat -Value Start-Chat
Set-Alias -Name chatbot -Value Start-Chat

Write-Host "💬 Natural chat mode loaded! Type " -NoNewline -ForegroundColor Green
Write-Host "chat" -NoNewline -ForegroundColor Cyan
Write-Host " for interactive session" -ForegroundColor Green
