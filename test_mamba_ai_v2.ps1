# ╔══════════════════════════════════════════════════════════════════╗
# ║         🐍 MAMBA AI V2 - COMPREHENSIVE TEST SUITE 🐍            ║
# ╚══════════════════════════════════════════════════════════════════╝

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
Write-Host "║         🧪 MAMBA AI V2 - SYSTEM TEST SUITE 🧪                   ║" -ForegroundColor Yellow
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
Write-Host ""

$testResults = @{
    Passed = 0
    Failed = 0
    Warnings = 0
    Tests = @()
}

function Test-Feature {
    param(
        [string]$Name,
        [scriptblock]$TestBlock
    )

    Write-Host "`n🧪 Testing: $Name" -ForegroundColor Cyan
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

    try {
        $result = & $TestBlock
        if ($result -eq $true) {
            Write-Host "✅ PASS: $Name" -ForegroundColor Green
            $testResults.Passed++
            $testResults.Tests += @{Name=$Name; Status='PASS'}
        } elseif ($result -eq 'WARN') {
            Write-Host "⚠️  WARN: $Name" -ForegroundColor Yellow
            $testResults.Warnings++
            $testResults.Tests += @{Name=$Name; Status='WARN'}
        } else {
            Write-Host "❌ FAIL: $Name" -ForegroundColor Red
            $testResults.Failed++
            $testResults.Tests += @{Name=$Name; Status='FAIL'}
        }
    } catch {
        Write-Host "❌ FAIL: $Name - $_" -ForegroundColor Red
        $testResults.Failed++
        $testResults.Tests += @{Name=$Name; Status='FAIL'; Error=$_}
    }
}

# ══════════════════════════════════════════════════════════════════
# TEST 1: Module Loading
# ══════════════════════════════════════════════════════════════════

Test-Feature -Name "Mamba AI v2 Module Loaded" -TestBlock {
    $commands = Get-Command -Name "ai", "ai-chat", "ai-code", "ai-fix", "ai-review" -ErrorAction SilentlyContinue
    if ($commands.Count -eq 5) {
        Write-Host "   Found all core AI commands" -ForegroundColor Gray
        return $true
    } else {
        Write-Host "   Missing commands. Found: $($commands.Count)/5" -ForegroundColor Red
        return $false
    }
}

# ══════════════════════════════════════════════════════════════════
# TEST 2: State Directory
# ══════════════════════════════════════════════════════════════════

Test-Feature -Name "State Directory Exists" -TestBlock {
    $mambaHome = "$env:USERPROFILE\.mamba"
    if (Test-Path $mambaHome) {
        Write-Host "   State directory: $mambaHome" -ForegroundColor Gray
        return $true
    } else {
        Write-Host "   State directory not found: $mambaHome" -ForegroundColor Red
        return $false
    }
}

# ══════════════════════════════════════════════════════════════════
# TEST 3: Conversation History
# ══════════════════════════════════════════════════════════════════

Test-Feature -Name "Conversation History Persistence" -TestBlock {
    $historyFile = "$env:USERPROFILE\.mamba\conversation_history.json"
    if (Test-Path $historyFile) {
        try {
            $history = Get-Content $historyFile -Raw | ConvertFrom-Json
            Write-Host "   History file exists with $($history.Count) messages" -ForegroundColor Gray
            return $true
        } catch {
            Write-Host "   History file corrupted" -ForegroundColor Red
            return $false
        }
    } else {
        Write-Host "   History file will be created on first use" -ForegroundColor Yellow
        return 'WARN'
    }
}

# ══════════════════════════════════════════════════════════════════
# TEST 4: API Keys
# ══════════════════════════════════════════════════════════════════

Test-Feature -Name "OpenAI API Key Configured" -TestBlock {
    if ($env:OPENAI_API_KEY) {
        $keyPreview = $env:OPENAI_API_KEY.Substring(0, [Math]::Min(10, $env:OPENAI_API_KEY.Length)) + "..."
        Write-Host "   OpenAI Key: $keyPreview" -ForegroundColor Gray
        return $true
    } else {
        Write-Host "   OpenAI API key not set. Set with: `$env:OPENAI_API_KEY = 'your-key'" -ForegroundColor Yellow
        return 'WARN'
    }
}

Test-Feature -Name "Claude API Key Configured (Optional)" -TestBlock {
    if ($env:ANTHROPIC_API_KEY) {
        Write-Host "   Claude Key: Configured ✓" -ForegroundColor Gray
        return $true
    } else {
        Write-Host "   Claude API key not set (optional)" -ForegroundColor Yellow
        return 'WARN'
    }
}

# ══════════════════════════════════════════════════════════════════
# TEST 5: Universal Codebase Detection
# ══════════════════════════════════════════════════════════════════

Test-Feature -Name "Current Codebase Detection" -TestBlock {
    $currentPath = Get-CurrentCodebase
    if ($currentPath) {
        Write-Host "   Current: $currentPath" -ForegroundColor Gray
        return $true
    } else {
        return $false
    }
}

Test-Feature -Name "Language Auto-Detection" -TestBlock {
    $language = Get-CodebaseLanguage -Path (Get-CurrentCodebase)
    if ($language) {
        Write-Host "   Detected: $language" -ForegroundColor Gray
        return $true
    } else {
        return $false
    }
}

# ══════════════════════════════════════════════════════════════════
# TEST 6: Security Sandbox
# ══════════════════════════════════════════════════════════════════

Test-Feature -Name "Security Sandbox - Block System Paths" -TestBlock {
    $result = Test-SafePath -Path "C:\Windows\System32"
    if ($result -eq $false) {
        Write-Host "   System32 correctly blocked ✓" -ForegroundColor Gray
        return $true
    } else {
        Write-Host "   Security sandbox not working!" -ForegroundColor Red
        return $false
    }
}

Test-Feature -Name "Security Sandbox - Allow User Paths" -TestBlock {
    $result = Test-SafePath -Path $env:USERPROFILE
    if ($result -eq $true) {
        Write-Host "   User directory correctly allowed ✓" -ForegroundColor Gray
        return $true
    } else {
        Write-Host "   User directory blocked incorrectly!" -ForegroundColor Red
        return $false
    }
}

# ══════════════════════════════════════════════════════════════════
# TEST 7: File System Tools
# ══════════════════════════════════════════════════════════════════

Test-Feature -Name "File System - Create Test File" -TestBlock {
    $testFile = "$env:TEMP\mamba_test_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
    $testContent = "Mamba AI v2 Test File - $(Get-Date)"

    Write-CodeFile -FilePath $testFile -Content $testContent

    if (Test-Path $testFile) {
        $readContent = Read-CodeFile -FilePath $testFile
        if ($readContent -eq $testContent) {
            Write-Host "   File write/read successful ✓" -ForegroundColor Gray
            Remove-Item $testFile -Force
            return $true
        } else {
            return $false
        }
    } else {
        return $false
    }
}

Test-Feature -Name "File System - Search Codebase" -TestBlock {
    $result = Search-Codebase -Pattern "function" -Path (Get-CurrentCodebase)
    if ($result) {
        Write-Host "   Search completed successfully ✓" -ForegroundColor Gray
        return $true
    } else {
        Write-Host "   No matches found (may be expected)" -ForegroundColor Yellow
        return 'WARN'
    }
}

# ══════════════════════════════════════════════════════════════════
# TEST 8: GitHub Tool
# ══════════════════════════════════════════════════════════════════

Test-Feature -Name "GitHub - Search Repos" -TestBlock {
    Write-Host "   Searching GitHub for 'backtesting python'..." -ForegroundColor Gray

    try {
        $repos = Search-GitHubRepos -Query "backtesting" -Language "Python" -MinStars 50

        if ($repos.Count -gt 0) {
            Write-Host "   Found $($repos.Count) repos ✓" -ForegroundColor Gray
            Write-Host "   Top repo: $($repos[0].Name) ⭐ $($repos[0].Stars)" -ForegroundColor Gray
            return $true
        } else {
            Write-Host "   No repos found" -ForegroundColor Yellow
            return 'WARN'
        }
    } catch {
        Write-Host "   GitHub API error (rate limit?): $_" -ForegroundColor Yellow
        return 'WARN'
    }
}

# ══════════════════════════════════════════════════════════════════
# TEST 9: Stack Overflow Tool
# ══════════════════════════════════════════════════════════════════

Test-Feature -Name "Stack Overflow - Search Questions" -TestBlock {
    Write-Host "   Searching Stack Overflow for 'python async'..." -ForegroundColor Gray

    try {
        $questions = Search-StackOverflow -Query "python async" -Tag "python"

        if ($questions.Count -gt 0) {
            Write-Host "   Found $($questions.Count) questions ✓" -ForegroundColor Gray
            Write-Host "   Top: $($questions[0].Title)" -ForegroundColor Gray
            return $true
        } else {
            Write-Host "   No questions found" -ForegroundColor Yellow
            return 'WARN'
        }
    } catch {
        Write-Host "   Stack Overflow API error: $_" -ForegroundColor Yellow
        return 'WARN'
    }
}

# ══════════════════════════════════════════════════════════════════
# TEST 10: Code Analysis Tool
# ══════════════════════════════════════════════════════════════════

Test-Feature -Name "Code Analysis - Create & Analyze Test Python File" -TestBlock {
    $testFile = "$env:TEMP\mamba_test.py"
    $testCode = @"
import os
import sys

def hello_world():
    print("Hello from Mamba AI v2!")
    return True

class TestClass:
    def __init__(self):
        self.name = "Mamba"

if __name__ == "__main__":
    hello_world()
"@

    Write-CodeFile -FilePath $testFile -Content $testCode

    if (Test-Path $testFile) {
        $analysis = Analyze-PythonCode -FilePath $testFile

        $valid = $analysis.SyntaxValid -and
                 $analysis.Functions.Count -eq 1 -and
                 $analysis.Classes.Count -eq 1 -and
                 $analysis.Imports.Count -eq 2

        Remove-Item $testFile -Force

        if ($valid) {
            Write-Host "   Syntax: $($analysis.SyntaxValid)" -ForegroundColor Gray
            Write-Host "   Functions: $($analysis.Functions.Count)" -ForegroundColor Gray
            Write-Host "   Classes: $($analysis.Classes.Count)" -ForegroundColor Gray
            Write-Host "   Imports: $($analysis.Imports.Count)" -ForegroundColor Gray
            return $true
        } else {
            return $false
        }
    } else {
        return $false
    }
}

# ══════════════════════════════════════════════════════════════════
# TEST 11: Python Tools (Optional)
# ══════════════════════════════════════════════════════════════════

Test-Feature -Name "Python - autopep8 Installed" -TestBlock {
    try {
        $result = python -m autopep8 --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   autopep8: $result" -ForegroundColor Gray
            return $true
        } else {
            Write-Host "   Install with: pip install autopep8" -ForegroundColor Yellow
            return 'WARN'
        }
    } catch {
        Write-Host "   Install with: pip install autopep8" -ForegroundColor Yellow
        return 'WARN'
    }
}

Test-Feature -Name "Python - isort Installed" -TestBlock {
    try {
        $result = python -m isort --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   isort: $result" -ForegroundColor Gray
            return $true
        } else {
            Write-Host "   Install with: pip install isort" -ForegroundColor Yellow
            return 'WARN'
        }
    } catch {
        Write-Host "   Install with: pip install isort" -ForegroundColor Yellow
        return 'WARN'
    }
}

# ══════════════════════════════════════════════════════════════════
# TEST 12: Logging System
# ══════════════════════════════════════════════════════════════════

Test-Feature -Name "Logging System" -TestBlock {
    $logFile = "$env:USERPROFILE\.mamba\mamba.log"

    Write-MambaLog -Level 'INFO' -Message "Test log entry" -Data @{test='successful'}

    if (Test-Path $logFile) {
        $lastLine = Get-Content $logFile -Tail 1
        if ($lastLine -match "Test log entry") {
            Write-Host "   Log file: $logFile ✓" -ForegroundColor Gray
            return $true
        } else {
            return $false
        }
    } else {
        return $false
    }
}

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
Write-Host "║                     📊 TEST SUMMARY 📊                          ║" -ForegroundColor Yellow
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
Write-Host ""

Write-Host "✅ Passed:   $($testResults.Passed)" -ForegroundColor Green
Write-Host "❌ Failed:   $($testResults.Failed)" -ForegroundColor $(if ($testResults.Failed -gt 0) { 'Red' } else { 'Gray' })
Write-Host "⚠️  Warnings: $($testResults.Warnings)" -ForegroundColor Yellow

$totalTests = $testResults.Passed + $testResults.Failed + $testResults.Warnings
$passRate = if ($totalTests -gt 0) { [math]::Round(($testResults.Passed / $totalTests) * 100, 1) } else { 0 }

Write-Host "`nPass Rate: $passRate%" -ForegroundColor $(if ($passRate -ge 80) { 'Green' } elseif ($passRate -ge 60) { 'Yellow' } else { 'Red' })

if ($testResults.Failed -eq 0) {
    Write-Host "`n🎉 All critical tests passed! Mamba AI v2 is ready to use!" -ForegroundColor Green
} else {
    Write-Host "`n⚠️  Some tests failed. Review errors above." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Set API keys if needed: `$env:OPENAI_API_KEY = 'your-key'" -ForegroundColor Gray
Write-Host "  2. Install Python tools: pip install autopep8 isort" -ForegroundColor Gray
Write-Host "  3. Try it out: ai 'explain this codebase'" -ForegroundColor Gray
Write-Host "  4. Read documentation: MAMBA_AI_V2_README.md" -ForegroundColor Gray
Write-Host ""
