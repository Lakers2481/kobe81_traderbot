# ╔══════════════════════════════════════════════════════════════════╗
# ║    🐍 MAMBA AI - SIMPLE CHAT INTERFACE 🐍                       ║
# ╚══════════════════════════════════════════════════════════════════╝

# Simple function - just talk naturally
function talk {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Message)

    if (-not $Message) {
        Write-Host ""
        Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
        Write-Host "║           🐍 MAMBA AI - READY TO HELP 🐍                        ║" -ForegroundColor Cyan
        Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Just type: " -NoNewline -ForegroundColor Yellow
        Write-Host "talk " -NoNewline -ForegroundColor Cyan
        Write-Host "followed by your question" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Examples:" -ForegroundColor Gray
        Write-Host "  talk what is this codebase?" -ForegroundColor White
        Write-Host "  talk find all bugs" -ForegroundColor White
        Write-Host "  talk explain this code" -ForegroundColor White
        Write-Host "  talk how does authentication work?" -ForegroundColor White
        Write-Host ""
        return
    }

    $question = $Message -join ' '

    Write-Host ""
    Write-Host "🐍 You: " -NoNewline -ForegroundColor Yellow
    Write-Host $question -ForegroundColor White
    Write-Host ""

    # Call the AI
    ai $question
}

# Quick alias
Set-Alias -Name ask -Value talk
Set-Alias -Name chat -Value talk

Write-Host "💬 Simple chat loaded! Type " -NoNewline -ForegroundColor Green
Write-Host "talk" -NoNewline -ForegroundColor Cyan
Write-Host " to start chatting" -ForegroundColor Green
