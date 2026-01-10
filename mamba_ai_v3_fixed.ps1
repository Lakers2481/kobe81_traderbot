# ╔══════════════════════════════════════════════════════════════════╗
# ║    🐍 MAMBA AI V3 - TRUE AUTONOMOUS INTELLIGENCE SYSTEM 🐍      ║
# ║        ReAct Loops • Self-Correction • Large Codebases          ║
# ╚══════════════════════════════════════════════════════════════════╝

# Load base v2 functionality
. "$PSScriptRoot\mamba_ai_v2.ps1"

# ══════════════════════════════════════════════════════════════════
# V3 CONFIGURATION
# ══════════════════════════════════════════════════════════════════

$script:V3_CONFIG = @{
    MaxIterations = 10
    MaxCodebaseFiles = 500
    ChunkSize = 50
    VerificationEnabled = $true
    SelfCorrectionEnabled = $true
    MaxRetries = 3
    IssueDetectionEnabled = $true
}

# ══════════════════════════════════════════════════════════════════
# REACT LOOP ENGINE
# ══════════════════════════════════════════════════════════════════

function Invoke-ReActLoop {
    param(
        [Parameter(Mandatory=$true)]
        [string]$UserQuery,
        [int]$MaxIterations = 10,
        [switch]$Verbose
    )

    Write-Host "`n╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║           🧠 MAMBA AI V3 - AUTONOMOUS MODE 🧠                   ║" -ForegroundColor Yellow
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📋 Task: $UserQuery" -ForegroundColor Cyan
    Write-Host "🔄 Max Iterations: $MaxIterations" -ForegroundColor Gray
    Write-Host ""

    $conversationHistory = @()
    $observations = @()
    $actionsCompleted = @()

    for ($i = 1; $i -le $MaxIterations; $i++) {
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
        Write-Host "🔄 Iteration $i / $MaxIterations" -ForegroundColor Yellow
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

        Write-Host "`n💭 THINKING..." -ForegroundColor Cyan

        $thinkPrompt = @"
USER TASK: $UserQuery

CONVERSATION HISTORY:
$($conversationHistory | ForEach-Object { "[$($_.Role)]: $($_.Content)" } | Out-String)

OBSERVATIONS FROM PREVIOUS ACTIONS:
$($observations | ConvertTo-Json -Depth 5)

ACTIONS COMPLETED SO FAR:
$($actionsCompleted | Out-String)

Think step-by-step:
1. What have I learned so far?
2. What do I still need to find out or do?
3. What tool/action should I use next?
4. Am I done, or do I need more information?

IMPORTANT: Respond ONLY with valid JSON:
{
  "reasoning": "your detailed thought process",
  "status": "continue" or "complete",
  "next_action": "tool_name",
  "tool_params": {"param1": "value1"},
  "confidence": 85
}

Available tools:
- read_file (params: file_path)
- write_file (params: file_path, content)
- search_codebase (params: pattern)
- analyze_code (params: file_path)
- search_github (params: query)
- get_codebase_context (params: none)
"@

        try {
            $thinkResponse = Invoke-MambaAI -Prompt $thinkPrompt
            $jsonMatch = $thinkResponse -match '\{[\s\S]*\}'

            if ($jsonMatch) {
                $jsonText = $Matches[0]
                $action = $jsonText | ConvertFrom-Json
            } else {
                throw "No valid JSON in response"
            }

        } catch {
            Write-Host "⚠️  Failed to parse AI response, retrying..." -ForegroundColor Yellow
            continue
        }

        if ($Verbose) {
            Write-Host "`n🧠 Reasoning:" -ForegroundColor Gray
            Write-Host $action.reasoning -ForegroundColor DarkGray
        }

        $conversationHistory += @{Role='think'; Content=$action.reasoning}

        if ($action.status -eq 'complete') {
            Write-Host "`n✅ TASK COMPLETE!" -ForegroundColor Green
            Write-Host "📝 Final Reasoning: $($action.reasoning)" -ForegroundColor Cyan

            return @{
                Status = 'Complete'
                Iterations = $i
                ActionsCompleted = $actionsCompleted
                FinalReasoning = $action.reasoning
                Observations = $observations
            }
        }

        Write-Host "`n🔧 ACTION: $($action.next_action)" -ForegroundColor Yellow
        Write-Host "📊 Confidence: $($action.confidence)%" -ForegroundColor Gray

        $observation = $null
        $actionSuccess = $true

        try {
            $observation = Invoke-Tool -ToolName $action.next_action -Params $action.tool_params
            $actionsCompleted += "$($action.next_action) completed"
        } catch {
            Write-Host "❌ Action failed: $_" -ForegroundColor Red
            $observation = "ERROR: $_"
            $actionSuccess = $false
        }

        Write-Host "`n👁️  OBSERVATION:" -ForegroundColor Magenta
        $observationPreview = if ($observation.Length -gt 300) {
            $observation.Substring(0, 300) + "..."
        } else {
            $observation
        }
        Write-Host $observationPreview -ForegroundColor DarkGray

        $observations += @{
            Iteration = $i
            Tool = $action.next_action
            Result = $observation
            Success = $actionSuccess
        }

        $conversationHistory += @{Role='observe'; Content=$observation}
        Write-Host ""
    }

    Write-Host "`n⚠️  Max iterations reached" -ForegroundColor Yellow
    return @{
        Status = 'MaxIterations'
        Iterations = $MaxIterations
        ActionsCompleted = $actionsCompleted
        Observations = $observations
    }
}

# ══════════════════════════════════════════════════════════════════
# TOOL SYSTEM
# ══════════════════════════════════════════════════════════════════

function Invoke-Tool {
    param(
        [Parameter(Mandatory=$true)]
        [string]$ToolName,
        [Parameter(Mandatory=$true)]
        [hashtable]$Params
    )

    switch ($ToolName) {
        'read_file' {
            return Read-CodeFile -FilePath $Params.file_path
        }
        'write_file' {
            return Write-CodeFile -FilePath $Params.file_path -Content $Params.content
        }
        'search_codebase' {
            return Search-Codebase -Pattern $Params.pattern
        }
        'analyze_code' {
            $analysis = Analyze-CodeFile -FilePath $Params.file_path
            return $analysis | ConvertTo-Json -Depth 10
        }
        'search_github' {
            $repos = Search-GitHubRepos -Query $Params.query
            return $repos | ConvertTo-Json -Depth 10
        }
        'get_codebase_context' {
            return Get-CodebaseContext
        }
        default {
            throw "Unknown tool: $ToolName"
        }
    }
}

# ══════════════════════════════════════════════════════════════════
# ISSUE DETECTION
# ══════════════════════════════════════════════════════════════════

function Find-CodebaseIssues {
    param([string]$Path = (Get-CurrentCodebase))

    Write-Host "`n╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║            🔍 CODEBASE ISSUE DETECTION 🔍                        ║" -ForegroundColor Yellow
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""

    $issues = @()
    $files = Get-ChildItem -Path $Path -Filter "*.py" -Recurse -File -ErrorAction SilentlyContinue

    Write-Host "📂 Scanning $($files.Count) Python files..." -ForegroundColor Cyan

    foreach ($file in $files) {
        Write-Host "   Checking: $($file.Name)" -ForegroundColor Gray

        try {
            $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue

            # Check 1: Syntax errors
            $analysis = Analyze-PythonCode -FilePath $file.FullName
            if ($analysis.Errors.Count -gt 0) {
                $issues += @{
                    File = $file.FullName
                    Type = 'SyntaxError'
                    Severity = 'High'
                    Description = $analysis.Errors -join "; "
                    Fix = "Review and fix syntax errors"
                }
            }

            # Check 2: TODO comments
            if ($content -match '(TODO|FIXME)') {
                $issues += @{
                    File = $file.FullName
                    Type = 'UnfinishedWork'
                    Severity = 'Medium'
                    Description = "Contains TODO/FIXME comments"
                    Fix = "Complete or remove TODO items"
                }
            }

            # Check 3: Hardcoded credentials
            if ($content -match 'password.*=|api_key.*=|secret.*=') {
                $issues += @{
                    File = $file.FullName
                    Type = 'SecurityRisk'
                    Severity = 'Critical'
                    Description = "Possible hardcoded credentials"
                    Fix = "Move to environment variables"
                }
            }

        } catch {
            # Skip files that can't be read
        }
    }

    # Display results
    Write-Host "`n📊 ISSUE SUMMARY:" -ForegroundColor Yellow
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

    $critical = ($issues | Where-Object { $_.Severity -eq 'Critical' }).Count
    $high = ($issues | Where-Object { $_.Severity -eq 'High' }).Count
    $medium = ($issues | Where-Object { $_.Severity -eq 'Medium' }).Count

    Write-Host "🔴 Critical: $critical" -ForegroundColor Red
    Write-Host "🟠 High:     $high" -ForegroundColor Yellow
    Write-Host "🟡 Medium:   $medium" -ForegroundColor Yellow
    Write-Host ""

    if ($issues.Count -eq 0) {
        Write-Host "✅ No issues found!" -ForegroundColor Green
    } else {
        Write-Host "`nDETAILED ISSUES:" -ForegroundColor Cyan
        foreach ($issue in $issues | Select-Object -First 20) {
            $severityColor = switch ($issue.Severity) {
                'Critical' { 'Red' }
                'High' { 'Yellow' }
                'Medium' { 'Yellow' }
                default { 'Gray' }
            }

            Write-Host "`n[$($issue.Severity)] $($issue.Type)" -ForegroundColor $severityColor
            Write-Host "  File: $($issue.File)" -ForegroundColor Gray
            Write-Host "  Issue: $($issue.Description)" -ForegroundColor Gray
            Write-Host "  Fix: $($issue.Fix)" -ForegroundColor Cyan
        }
    }

    return $issues
}

# ══════════════════════════════════════════════════════════════════
# HIGH-LEVEL V3 COMMANDS
# ══════════════════════════════════════════════════════════════════

function ai-autonomous {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Task
    )

    $result = Invoke-ReActLoop -UserQuery $Task -MaxIterations $script:V3_CONFIG.MaxIterations -Verbose

    Write-Host "`n╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║                    📋 FINAL SUMMARY 📋                          ║" -ForegroundColor Green
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Write-Host "Status: $($result.Status)" -ForegroundColor Cyan
    Write-Host "Iterations: $($result.Iterations)" -ForegroundColor Gray
    Write-Host ""

    return $result
}

function ai-scan-issues {
    $issues = Find-CodebaseIssues
    $reportPath = Join-Path (Get-CurrentCodebase) "mamba_issue_report.json"
    $issues | ConvertTo-Json -Depth 10 | Set-Content $reportPath
    Write-Host "`n📄 Report saved: $reportPath" -ForegroundColor Cyan
    return $issues
}

# ══════════════════════════════════════════════════════════════════
# INITIALIZATION
# ══════════════════════════════════════════════════════════════════

Write-Host ""
Write-Host "🐍 Mamba AI v3 loaded! Type " -NoNewline -ForegroundColor Yellow
Write-Host "ai-autonomous" -NoNewline -ForegroundColor Cyan
Write-Host " for autonomous mode" -ForegroundColor Yellow
Write-Host ""

# Functions are available when dot-sourced (no Export-ModuleMember needed)
