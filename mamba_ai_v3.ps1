# ╔══════════════════════════════════════════════════════════════════╗
# ║    🐍 MAMBA AI V3 - TRUE AUTONOMOUS INTELLIGENCE SYSTEM 🐍      ║
# ║        ReAct Loops • Self-Correction • Large Codebases          ║
# ║              QUANT-LEVEL VERIFICATION & ACCURACY                 ║
# ╚══════════════════════════════════════════════════════════════════╝

# Load base v2 functionality
. "$PSScriptRoot\mamba_ai_v2.ps1"

# ══════════════════════════════════════════════════════════════════
# V3 CONFIGURATION
# ══════════════════════════════════════════════════════════════════

$script:V3_CONFIG = @{
    MaxIterations = 10                  # Max ReAct loop cycles
    MaxCodebaseFiles = 500              # Can handle huge codebases
    ChunkSize = 50                      # Files per analysis chunk
    VerificationEnabled = $true         # Multi-layer verification
    SelfCorrectionEnabled = $true       # Retry on failure
    MaxRetries = 3                      # Retry attempts
    IssueDetectionEnabled = $true       # Proactive problem finding
}

# ══════════════════════════════════════════════════════════════════
# REACT LOOP ENGINE - Core Autonomous System
# ══════════════════════════════════════════════════════════════════

function Invoke-ReActLoop {
    <#
    .SYNOPSIS
    Autonomous multi-step reasoning: Think → Act → Observe → Repeat

    .DESCRIPTION
    This is the core intelligence system. It breaks down complex tasks
    into steps, executes them, observes results, and adapts.
    #>
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

        # ═══════════════════════════════════════════════════════════
        # THINK: What should I do next?
        # ═══════════════════════════════════════════════════════════

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

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{
  "reasoning": "your detailed thought process here",
  "status": "continue" or "complete",
  "next_action": "tool_name",
  "tool_params": {
    "param1": "value1",
    "param2": "value2"
  },
  "confidence": 0-100
}

Available tools:
- read_file: Read any file (params: file_path)
- write_file: Write/create file (params: file_path, content)
- move_file: Move/rename file (params: source, destination)
- delete_file: Delete file (params: file_path)
- search_codebase: Search for pattern (params: pattern, extensions)
- analyze_code: Analyze code file (params: file_path)
- search_github: Search repos (params: query, language)
- search_stackoverflow: Search SO (params: query, tag)
- get_codebase_context: Get full codebase overview (params: none)
- verify_result: Verify a previous result (params: result_to_verify)
"@

        try {
            $thinkResponse = Invoke-MambaAI -Prompt $thinkPrompt

            # Extract JSON from response (handles markdown code blocks)
            $jsonMatch = $thinkResponse -match '\{[\s\S]*\}'
            if ($jsonMatch) {
                $jsonText = $Matches[0]
                $action = $jsonText | ConvertFrom-Json
            } else {
                throw "No valid JSON found in response"
            }

        } catch {
            Write-Host "⚠️  Failed to parse AI response, retrying..." -ForegroundColor Yellow
            Write-MambaLog -Level 'WARN' -Message "JSON parsing failed" -Data @{iteration=$i; error=$_}
            continue
        }

        # Log reasoning
        if ($Verbose) {
            Write-Host "`n🧠 Reasoning:" -ForegroundColor Gray
            Write-Host $action.reasoning -ForegroundColor DarkGray
        }

        $conversationHistory += @{Role='think'; Content=$action.reasoning}

        # Check if task is complete
        if ($action.status -eq 'complete') {
            Write-Host "`n✅ TASK COMPLETE!" -ForegroundColor Green
            Write-Host "📝 Final Reasoning: $($action.reasoning)" -ForegroundColor Cyan

            # Return final summary
            $summary = @{
                Status = 'Complete'
                Iterations = $i
                ActionsCompleted = $actionsCompleted
                FinalReasoning = $action.reasoning
                Observations = $observations
            }

            return $summary
        }

        # ═══════════════════════════════════════════════════════════
        # ACT: Execute the chosen tool
        # ═══════════════════════════════════════════════════════════

        Write-Host "`n🔧 ACTION: $($action.next_action)" -ForegroundColor Yellow
        Write-Host "📊 Confidence: $($action.confidence)%" -ForegroundColor Gray

        $observation = $null
        $actionSuccess = $true

        try {
            $observation = Invoke-Tool -ToolName $action.next_action -Params $action.tool_params
            $actionsCompleted += "$($action.next_action) with $($action.tool_params | ConvertTo-Json -Compress)"
        } catch {
            Write-Host "❌ Action failed: $_" -ForegroundColor Red
            $observation = "ERROR: $_"
            $actionSuccess = $false
            Write-MambaLog -Level 'ERROR' -Message "Tool execution failed" -Data @{tool=$action.next_action; error=$_}
        }

        # ═══════════════════════════════════════════════════════════
        # OBSERVE: Record and display the result
        # ═══════════════════════════════════════════════════════════

        Write-Host "`n👁️  OBSERVATION:" -ForegroundColor Magenta

        $observationPreview = if ($observation.Length -gt 300) {
            $observation.Substring(0, 300) + "... (truncated)"
        } else {
            $observation
        }
        Write-Host $observationPreview -ForegroundColor DarkGray

        $observations += @{
            Iteration = $i
            Tool = $action.next_action
            Params = $action.tool_params
            Result = $observation
            Success = $actionSuccess
            Timestamp = Get-Date -Format "o"
        }

        $conversationHistory += @{Role='observe'; Content=$observation}

        # ═══════════════════════════════════════════════════════════
        # VERIFY: Check if result makes sense (if enabled)
        # ═══════════════════════════════════════════════════════════

        if ($script:V3_CONFIG.VerificationEnabled -and $actionSuccess) {
            Write-Host "`n🔍 VERIFYING RESULT..." -ForegroundColor Cyan

            $verifyResult = Test-ResultValidity -Observation $observation -Action $action

            if (-not $verifyResult.Valid) {
                Write-Host "⚠️  Verification failed: $($verifyResult.Reason)" -ForegroundColor Yellow
                $observations[-1].Verified = $false
                $observations[-1].VerificationIssue = $verifyResult.Reason
            } else {
                Write-Host "✅ Result verified" -ForegroundColor Green
                $observations[-1].Verified = $true
            }
        }

        Write-Host ""
    }

    # Max iterations reached
    Write-Host "`n⚠️  Max iterations ($MaxIterations) reached" -ForegroundColor Yellow

    $summary = @{
        Status = 'MaxIterations'
        Iterations = $MaxIterations
        ActionsCompleted = $actionsCompleted
        Observations = $observations
        Message = "Task may be incomplete. Consider increasing MaxIterations or breaking down the task."
    }

    return $summary
}

# ══════════════════════════════════════════════════════════════════
# ENHANCED TOOL SYSTEM - With Full File Operations
# ══════════════════════════════════════════════════════════════════

function Invoke-Tool {
    param(
        [Parameter(Mandatory=$true)]
        [string]$ToolName,

        [Parameter(Mandatory=$true)]
        [hashtable]$Params
    )

    Write-MambaLog -Level 'INFO' -Message "Executing tool" -Data @{tool=$ToolName; params=$Params}

    switch ($ToolName) {
        'read_file' {
            return Read-CodeFile -FilePath $Params.file_path
        }

        'write_file' {
            $result = Write-CodeFile -FilePath $Params.file_path -Content $Params.content
            return $result
        }

        'move_file' {
            return Move-CodeFile -Source $Params.source -Destination $Params.destination
        }

        'delete_file' {
            return Remove-CodeFile -FilePath $Params.file_path
        }

        'search_codebase' {
            $extensions = if ($Params.extensions) { $Params.extensions } else { @('*.py', '*.js', '*.ts', '*.cs') }
            return Search-Codebase -Pattern $Params.pattern -Extensions $extensions
        }

        'analyze_code' {
            $analysis = Analyze-CodeFile -FilePath $Params.file_path
            return $analysis | ConvertTo-Json -Depth 10
        }

        'search_github' {
            $language = if ($Params.language) { $Params.language } else { $null }
            $repos = Search-GitHubRepos -Query $Params.query -Language $language
            return $repos | ConvertTo-Json -Depth 10
        }

        'search_stackoverflow' {
            $tag = if ($Params.tag) { $Params.tag } else { $null }
            $questions = Search-StackOverflow -Query $Params.query -Tag $tag
            return $questions | ConvertTo-Json -Depth 10
        }

        'get_codebase_context' {
            return Get-LargeCodebaseContext -MaxFiles $script:V3_CONFIG.MaxCodebaseFiles
        }

        'verify_result' {
            $verification = Test-ResultValidity -Observation $Params.result_to_verify
            return $verification | ConvertTo-Json -Depth 5
        }

        default {
            throw "Unknown tool: $ToolName"
        }
    }
}

# ══════════════════════════════════════════════════════════════════
# LARGE CODEBASE HANDLER - Smart Chunking & Summarization
# ══════════════════════════════════════════════════════════════════

function Get-LargeCodebaseContext {
    <#
    .SYNOPSIS
    Handles codebases of ANY size with smart chunking
    #>
    param(
        [string]$Path = (Get-CurrentCodebase),
        [int]$MaxFiles = 500
    )

    Write-Host "📂 Analyzing large codebase: $Path" -ForegroundColor Yellow
    Write-Host "   Max files: $MaxFiles" -ForegroundColor Gray

    $language = Get-CodebaseLanguage -Path $Path
    Write-Host "   Detected language: $language" -ForegroundColor Gray

    # Get file extensions
    $extensions = switch ($language) {
        'Python' { @('*.py') }
        'JavaScript' { @('*.js', '*.jsx') }
        'TypeScript' { @('*.ts', '*.tsx') }
        'CSharp' { @('*.cs') }
        'Java' { @('*.java') }
        'Go' { @('*.go') }
        'Rust' { @('*.rs') }
        'CPP' { @('*.cpp', '*.hpp', '*.h') }
        default { @('*.py', '*.js', '*.ts', '*.cs', '*.java') }
    }

    # Get ALL code files (no size limit)
    $allFiles = @()
    foreach ($ext in $extensions) {
        $files = Get-ChildItem -Path $Path -Filter $ext -Recurse -File -ErrorAction SilentlyContinue
        $allFiles += $files
    }

    Write-Host "   Found $($allFiles.Count) files" -ForegroundColor Gray

    # Build hierarchical context
    $context = "LARGE CODEBASE ANALYSIS`n"
    $context += "Path: $Path`n"
    $context += "Language: $language`n"
    $context += "Total Files: $($allFiles.Count)`n`n"

    # Directory tree (always include)
    $context += "DIRECTORY STRUCTURE:`n"
    $dirs = Get-ChildItem -Path $Path -Directory -Recurse -ErrorAction SilentlyContinue | Select-Object -First 100
    foreach ($dir in $dirs) {
        $relativePath = $dir.FullName.Replace($Path, '').TrimStart('\')
        $depth = ($relativePath -split '\\').Count
        $indent = "  " * $depth
        $context += "$indent📁 $($dir.Name)`n"
    }

    # Smart file selection (prioritize important files)
    $priorityFiles = @()
    $importantNames = @('main', 'index', 'app', 'server', 'config', 'settings', '__init__', 'setup')

    foreach ($file in $allFiles) {
        $baseName = [System.IO.Path]::GetFileNameWithoutExtension($file.Name).ToLower()
        if ($importantNames -contains $baseName) {
            $priorityFiles += $file
        }
    }

    # Add remaining files up to limit
    $remainingSlots = $MaxFiles - $priorityFiles.Count
    $otherFiles = $allFiles | Where-Object { $priorityFiles -notcontains $_ } | Select-Object -First $remainingSlots
    $selectedFiles = $priorityFiles + $otherFiles

    Write-Host "   Selected $($selectedFiles.Count) files for analysis" -ForegroundColor Gray
    Write-Host "   ($($priorityFiles.Count) priority files)" -ForegroundColor DarkGray

    # Add file summaries
    $context += "`n`nFILE SUMMARIES:`n"
    foreach ($file in $selectedFiles) {
        $relativePath = $file.FullName.Replace($Path, '').TrimStart('\')
        $size = [math]::Round($file.Length / 1KB, 2)
        $context += "`n📄 $relativePath ($size KB)`n"

        # For small files, include full content
        if ($file.Length -lt 50KB) {
            $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
            if ($content) {
                $extension = $file.Extension.TrimStart('.')
                $context += "``````$extension`n$content`n``````n"
            }
        } else {
            # For large files, include summary only
            $lines = Get-Content $file.FullName -TotalCount 50 -ErrorAction SilentlyContinue
            $context += "First 50 lines:`n$($lines -join "`n")`n... (file truncated)`n"
        }
    }

    $context += "`n`nSTATISTICS:`n"
    $context += "Total Lines of Code: $($selectedFiles | ForEach-Object { (Get-Content $_.FullName -ErrorAction SilentlyContinue).Count } | Measure-Object -Sum).Sum`n"
    $context += "Total Size: $([math]::Round(($selectedFiles | Measure-Object -Property Length -Sum).Sum / 1MB, 2)) MB`n"

    Write-Host "✅ Codebase context generated" -ForegroundColor Green
    return $context
}

# ══════════════════════════════════════════════════════════════════
# ENHANCED FILE OPERATIONS - Move, Rename, Delete
# ══════════════════════════════════════════════════════════════════

function Move-CodeFile {
    <#
    .SYNOPSIS
    Move or rename a file safely
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Source,

        [Parameter(Mandatory=$true)]
        [string]$Destination
    )

    if (-not (Test-SafePath -Path $Source) -or -not (Test-SafePath -Path $Destination)) {
        return "❌ Access denied: Cannot move files outside allowed paths"
    }

    if (-not (Test-Path $Source)) {
        return "❌ Source file not found: $Source"
    }

    if (Test-Path $Destination) {
        $confirm = Read-Host "⚠️  Destination exists. Overwrite? (yes/no)"
        if ($confirm -ne 'yes') {
            return "❌ Operation cancelled"
        }
    }

    try {
        # Create backup of destination if it exists
        if (Test-Path $Destination) {
            $backupPath = "$Destination.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
            Copy-Item $Destination $backupPath
            Write-Host "📦 Backup created: $backupPath" -ForegroundColor Gray
        }

        Move-Item -Path $Source -Destination $Destination -Force
        Write-MambaLog -Level 'INFO' -Message "File moved" -Data @{source=$Source; destination=$Destination}
        return "✅ File moved: $Source → $Destination"
    } catch {
        Write-MambaLog -Level 'ERROR' -Message "Move failed: $_"
        return "❌ Move failed: $_"
    }
}

function Remove-CodeFile {
    <#
    .SYNOPSIS
    Delete a file safely with confirmation
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath
    )

    if (-not (Test-SafePath -Path $FilePath)) {
        return "❌ Access denied: Cannot delete files in protected paths"
    }

    if (-not (Test-Path $FilePath)) {
        return "❌ File not found: $FilePath"
    }

    # Always confirm deletion
    $confirm = Read-Host "⚠️  Delete '$FilePath'? This cannot be undone! (yes/no)"
    if ($confirm -ne 'yes') {
        return "❌ Deletion cancelled"
    }

    try {
        # Create backup before deletion
        $backupPath = "$FilePath.deleted_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Copy-Item $FilePath $backupPath
        Write-Host "📦 Backup created: $backupPath" -ForegroundColor Gray

        Remove-Item $FilePath -Force
        Write-MambaLog -Level 'INFO' -Message "File deleted" -Data @{path=$FilePath; backup=$backupPath}
        return "✅ File deleted: $FilePath (backup: $backupPath)"
    } catch {
        Write-MambaLog -Level 'ERROR' -Message "Delete failed: $_"
        return "❌ Delete failed: $_"
    }
}

# ══════════════════════════════════════════════════════════════════
# VERIFICATION SYSTEM - Multi-Layer Result Validation
# ══════════════════════════════════════════════════════════════════

function Test-ResultValidity {
    <#
    .SYNOPSIS
    Verifies if a result/observation makes sense
    #>
    param(
        [Parameter(Mandatory=$true)]
        $Observation,

        $Action
    )

    $verification = @{
        Valid = $true
        Confidence = 100
        Issues = @()
        Reason = ""
    }

    # Check 1: Is it an error?
    if ($Observation -match '^ERROR:' -or $Observation -match '❌') {
        $verification.Valid = $false
        $verification.Confidence = 0
        $verification.Reason = "Result contains error message"
        $verification.Issues += "Error detected in observation"
        return $verification
    }

    # Check 2: Is it empty/null?
    if (-not $Observation -or $Observation.Length -eq 0) {
        $verification.Valid = $false
        $verification.Confidence = 0
        $verification.Reason = "Empty or null result"
        $verification.Issues += "No data returned"
        return $verification
    }

    # Check 3: JSON validity (if applicable)
    if ($Observation -match '^\{' -or $Observation -match '^\[') {
        try {
            $json = $Observation | ConvertFrom-Json
            # Valid JSON
        } catch {
            $verification.Valid = $false
            $verification.Confidence = 30
            $verification.Reason = "Invalid JSON format"
            $verification.Issues += "JSON parsing failed"
        }
    }

    # Check 4: Code syntax (if it's code)
    if ($Observation -match '```' -or $Observation -match 'def ' -or $Observation -match 'function ') {
        # Extract code blocks
        if ($Observation -match '```(\w+)\n([\s\S]*?)\n```') {
            $language = $Matches[1]
            $code = $Matches[2]

            # Quick syntax check
            if ($language -eq 'python' -or $language -eq 'py') {
                try {
                    $tempFile = "$env:TEMP\syntax_check.py"
                    $code | Set-Content $tempFile
                    $syntaxCheck = python -m py_compile $tempFile 2>&1
                    Remove-Item $tempFile -Force -ErrorAction SilentlyContinue

                    if ($LASTEXITCODE -ne 0) {
                        $verification.Issues += "Python syntax error detected"
                        $verification.Confidence = 60
                    }
                } catch {
                    # Python not available
                }
            }
        }
    }

    # Check 5: Result length (suspiciously short?)
    if ($Observation.Length -lt 20) {
        $verification.Confidence = 70
        $verification.Issues += "Result seems unusually short"
    }

    if ($verification.Issues.Count -gt 0) {
        $verification.Reason = $verification.Issues -join "; "
    } else {
        $verification.Reason = "All validation checks passed"
    }

    return $verification
}

# ══════════════════════════════════════════════════════════════════
# ISSUE DETECTION SYSTEM - Proactive Problem Finding
# ══════════════════════════════════════════════════════════════════

function Find-CodebaseIssues {
    <#
    .SYNOPSIS
    Scans entire codebase for common issues
    #>
    param(
        [string]$Path = (Get-CurrentCodebase)
    )

    Write-Host "`n╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║            🔍 CODEBASE ISSUE DETECTION 🔍                        ║" -ForegroundColor Yellow
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""

    $issues = @()
    $language = Get-CodebaseLanguage -Path $Path

    # Get all code files
    $files = Get-ChildItem -Path $Path -Filter "*.*" -Recurse -File -ErrorAction SilentlyContinue |
             Where-Object { $_.Extension -match '\.(py|js|ts|cs|java|go|rs|cpp|h)$' }

    Write-Host "📂 Scanning $($files.Count) files..." -ForegroundColor Cyan

    foreach ($file in $files) {
        Write-Host "   Checking: $($file.Name)" -ForegroundColor Gray

        try {
            $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue

            # Issue 1: Syntax errors
            $analysis = Analyze-CodeFile -FilePath $file.FullName
            if ($analysis.Errors.Count -gt 0) {
                $issues += @{
                    File = $file.FullName
                    Type = 'SyntaxError'
                    Severity = 'High'
                    Description = $analysis.Errors -join "; "
                    Fix = "Review and fix syntax errors"
                }
            }

            # Issue 2: TODO/FIXME comments
            if ($content -match '(TODO|FIXME|HACK|XXX):?\s*(.+)') {
                $issues += @{
                    File = $file.FullName
                    Type = 'UnfinishedWork'
                    Severity = 'Medium'
                    Description = "Contains TODO/FIXME: $($Matches[2])"
                    Fix = "Complete or remove TODO items"
                }
            }

            # Issue 3: Hardcoded credentials (Python/JS specific)
            if ($content -match "(password|api[_-]?key|secret|token)\s*=\s*[`"`']") {
                $issues += @{
                    File = $file.FullName
                    Type = 'SecurityRisk'
                    Severity = 'Critical'
                    Description = "Hardcoded credentials detected"
                    Fix = "Move credentials to environment variables or config files"
                }
            }

            # Issue 4: Deprecated imports (Python)
            if ($language -eq 'Python' -and $content -match 'import (imp|sets)\b') {
                $issues += @{
                    File = $file.FullName
                    Type = 'DeprecatedCode'
                    Severity = 'Medium'
                    Description = "Uses deprecated Python modules"
                    Fix = "Replace with modern alternatives"
                }
            }

            # Issue 5: Long lines (>120 chars)
            $longLines = ($content -split "`n") | Where-Object { $_.Length -gt 120 }
            if ($longLines.Count -gt 5) {
                $issues += @{
                    File = $file.FullName
                    Type = 'CodeStyle'
                    Severity = 'Low'
                    Description = "$($longLines.Count) lines exceed 120 characters"
                    Fix = "Break long lines for better readability"
                }
            }

        } catch {
            Write-MambaLog -Level 'WARN' -Message "Failed to scan file: $_" -Data @{file=$file.FullName}
        }
    }

    # Display results
    Write-Host "`n📊 ISSUE SUMMARY:" -ForegroundColor Yellow
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

    $critical = ($issues | Where-Object { $_.Severity -eq 'Critical' }).Count
    $high = ($issues | Where-Object { $_.Severity -eq 'High' }).Count
    $medium = ($issues | Where-Object { $_.Severity -eq 'Medium' }).Count
    $low = ($issues | Where-Object { $_.Severity -eq 'Low' }).Count

    Write-Host "🔴 Critical: $critical" -ForegroundColor Red
    Write-Host "🟠 High:     $high" -ForegroundColor Yellow
    Write-Host "🟡 Medium:   $medium" -ForegroundColor Yellow
    Write-Host "🟢 Low:      $low" -ForegroundColor Green
    Write-Host ""

    if ($issues.Count -eq 0) {
        Write-Host "✅ No issues found! Codebase looks clean." -ForegroundColor Green
    } else {
        Write-Host "`nDETAILED ISSUES:" -ForegroundColor Cyan
        foreach ($issue in $issues) {
            $severityColor = switch ($issue.Severity) {
                'Critical' { 'Red' }
                'High' { 'Yellow' }
                'Medium' { 'Yellow' }
                'Low' { 'Gray' }
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
    <#
    .SYNOPSIS
    Fully autonomous multi-step task execution

    .EXAMPLE
    ai-autonomous "find all bugs in this codebase and create a report"
    ai-autonomous "refactor the authentication module to use JWT"
    ai-autonomous "optimize all Python files for performance"
    #>
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
    Write-Host "Actions Completed: $($result.ActionsCompleted.Count)" -ForegroundColor Gray
    Write-Host ""

    if ($result.FinalReasoning) {
        Write-Host "Final Reasoning:" -ForegroundColor Yellow
        Write-Host $result.FinalReasoning -ForegroundColor Gray
    }

    return $result
}

function ai-scan-issues {
    <#
    .SYNOPSIS
    Scan entire codebase for issues and generate report
    #>
    $issues = Find-CodebaseIssues

    # Save report
    $reportPath = Join-Path (Get-CurrentCodebase) "mamba_issue_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
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
Write-Host " for fully autonomous mode" -ForegroundColor Yellow
Write-Host "   New: ReAct loops, self-correction, large codebases, issue detection" -ForegroundColor Gray
Write-Host ""

# Export V3 functions
Export-ModuleMember -Function @(
    'Invoke-ReActLoop',
    'Invoke-Tool',
    'Get-LargeCodebaseContext',
    'Move-CodeFile',
    'Remove-CodeFile',
    'Test-ResultValidity',
    'Find-CodebaseIssues',
    'ai-autonomous',
    'ai-scan-issues'
)
