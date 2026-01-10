# ╔══════════════════════════════════════════════════════════════════╗
# ║        🐍 MAMBA AI V2 - GENIUS-LEVEL UNIVERSAL ASSISTANT 🐍       ║
# ║              Beyond ChatGPT & Claude - Tool-Powered               ║
# ║                     QUANT-LEVEL ACCURACY                          ║
# ╚══════════════════════════════════════════════════════════════════╝

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

# Load API keys from environment
$script:OPENAI_API_KEY = $env:OPENAI_API_KEY
$script:ANTHROPIC_API_KEY = $env:ANTHROPIC_API_KEY
$script:GROQ_API_KEY = $env:GROQ_API_KEY
$script:YOUTUBE_API_KEY = $env:YOUTUBE_API_KEY

# Auto-detect which provider to use (prefer Groq > Claude > OpenAI)
if ($env:GROQ_API_KEY) {
    $script:DEFAULT_MODEL = "llama-3.1-70b-versatile"
} elseif ($env:ANTHROPIC_API_KEY) {
    $script:DEFAULT_MODEL = "claude-3-opus-20240229"
} elseif ($env:OPENAI_API_KEY) {
    $script:DEFAULT_MODEL = "gpt-4"
} else {
    $script:DEFAULT_MODEL = "llama-3.1-70b-versatile"  # Fallback to Groq
}

$script:CONVERSATION_HISTORY = @()

# State directories
$script:MAMBA_HOME = "$env:USERPROFILE\.mamba"
$script:HISTORY_FILE = "$script:MAMBA_HOME\conversation_history.json"
$script:LOG_FILE = "$script:MAMBA_HOME\mamba.log"
$script:CONFIG_FILE = "$script:MAMBA_HOME\config.json"

# Initialize directories
if (-not (Test-Path $script:MAMBA_HOME)) {
    New-Item -ItemType Directory -Path $script:MAMBA_HOME -Force | Out-Null
}

# ══════════════════════════════════════════════════════════════════
# SECURITY CONFIGURATION
# ══════════════════════════════════════════════════════════════════

$script:SECURITY_CONFIG = @{
    AllowedPaths = @(
        $env:USERPROFILE,
        'C:\Temp',
        'C:\Users'
    )
    BlockedPaths = @(
        'C:\Windows\System32',
        'C:\Program Files\WindowsPowerShell',
        'HKLM:\',
        'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Run'
    )
    DangerousCommands = @(
        'Remove-Item -Recurse -Force',
        'Format-Volume',
        'Stop-Process -Name explorer',
        'rm -rf',
        'del /F /S /Q'
    )
    RequireConfirmation = @(
        'Remove-Item',
        'Move-Item',
        'Rename-Item'
    )
}

function Test-SafePath {
    param([string]$Path)

    # Check if path is blocked
    foreach ($blocked in $script:SECURITY_CONFIG.BlockedPaths) {
        if ($Path -like "$blocked*") {
            Write-Host "❌ ACCESS DENIED: Path '$Path' is protected" -ForegroundColor Red
            return $false
        }
    }

    # Check if path is allowed
    $isAllowed = $false
    foreach ($allowed in $script:SECURITY_CONFIG.AllowedPaths) {
        if ($Path -like "$allowed*") {
            $isAllowed = $true
            break
        }
    }

    return $isAllowed
}

function Test-SafeOperation {
    param(
        [string]$Operation,
        [string]$Path
    )

    # Check for dangerous commands
    foreach ($dangerous in $script:SECURITY_CONFIG.DangerousCommands) {
        if ($Operation -like "*$dangerous*") {
            Write-Host "❌ OPERATION BLOCKED: Dangerous command detected" -ForegroundColor Red
            return $false
        }
    }

    # Require confirmation for destructive ops
    foreach ($requireConfirm in $script:SECURITY_CONFIG.RequireConfirmation) {
        if ($Operation -like "*$requireConfirm*") {
            $confirm = Read-Host "⚠️  Confirm operation: $Operation on '$Path'? (yes/no)"
            if ($confirm -ne 'yes') {
                Write-Host "❌ Operation cancelled by user" -ForegroundColor Yellow
                return $false
            }
        }
    }

    return Test-SafePath -Path $Path
}

# ══════════════════════════════════════════════════════════════════
# LOGGING & STATE PERSISTENCE
# ══════════════════════════════════════════════════════════════════

function Write-MambaLog {
    param(
        [string]$Level,  # INFO, WARN, ERROR
        [string]$Message,
        [hashtable]$Data = @{}
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "$timestamp [$Level] $Message"

    if ($Data.Count -gt 0) {
        $logEntry += " | " + ($Data | ConvertTo-Json -Compress)
    }

    Add-Content -Path $script:LOG_FILE -Value $logEntry -ErrorAction SilentlyContinue

    # Also print to console with color
    $color = switch ($Level) {
        'INFO' { 'Cyan' }
        'WARN' { 'Yellow' }
        'ERROR' { 'Red' }
        default { 'White' }
    }
    Write-Host $logEntry -ForegroundColor $color
}

function Save-ConversationHistory {
    param([array]$History)

    try {
        $History | ConvertTo-Json -Depth 10 | Set-Content $script:HISTORY_FILE
    } catch {
        Write-MambaLog -Level 'WARN' -Message "Failed to save conversation history: $_"
    }
}

function Load-ConversationHistory {
    if (Test-Path $script:HISTORY_FILE) {
        try {
            return Get-Content $script:HISTORY_FILE -Raw | ConvertFrom-Json
        } catch {
            Write-MambaLog -Level 'WARN' -Message "Failed to load conversation history: $_"
        }
    }
    return @()
}

function Add-ToHistory {
    param(
        [string]$Role,
        [string]$Content
    )

    try {
        # Ensure content is a string
        $contentStr = [string]$Content

        $script:CONVERSATION_HISTORY += @{
            role = $Role
            content = $contentStr
            timestamp = Get-Date -Format "o"
        }

        # Keep only last 100 messages
        if ($script:CONVERSATION_HISTORY.Count -gt 100) {
            $script:CONVERSATION_HISTORY = $script:CONVERSATION_HISTORY[-100..-1]
        }

        Save-ConversationHistory -History $script:CONVERSATION_HISTORY
    } catch {
        Write-MambaLog -Level 'WARN' -Message "Failed to add to history: $_"
    }
}

# Load conversation history on startup
$script:CONVERSATION_HISTORY = Load-ConversationHistory

# ══════════════════════════════════════════════════════════════════
# PHASE 1: UNIVERSAL FILE SYSTEM ACCESS
# ══════════════════════════════════════════════════════════════════

function Get-CurrentCodebase {
    <#
    .SYNOPSIS
    Returns current PowerShell working directory (universal codebase detection)
    #>
    return $PWD.Path
}

function Get-CodebaseLanguage {
    param([string]$Path)

    $indicators = @{
        'Python' = @('*.py', 'requirements.txt', 'setup.py', 'pyproject.toml', '.python-version')
        'JavaScript' = @('package.json', '*.js', '*.jsx', 'node_modules')
        'TypeScript' = @('tsconfig.json', '*.ts', '*.tsx')
        'CSharp' = @('*.csproj', '*.sln', '*.cs')
        'Java' = @('pom.xml', 'build.gradle', '*.java')
        'Go' = @('go.mod', 'go.sum', '*.go')
        'Rust' = @('Cargo.toml', 'Cargo.lock', '*.rs')
        'CPP' = @('CMakeLists.txt', 'Makefile', '*.cpp', '*.hpp', '*.h')
    }

    $scores = @{}
    foreach ($lang in $indicators.Keys) {
        $scores[$lang] = 0
        foreach ($pattern in $indicators[$lang]) {
            $files = Get-ChildItem -Path $Path -Filter $pattern -Recurse -ErrorAction SilentlyContinue | Select-Object -First 5
            $scores[$lang] += $files.Count
        }
    }

    # Return dominant language
    $dominant = ($scores.GetEnumerator() | Sort-Object -Property Value -Descending | Select-Object -First 1).Key
    return $dominant
}

function Get-CodebaseContext {
    param(
        [string]$Path = (Get-CurrentCodebase),
        [int]$MaxFiles = 50,
        [int]$MaxFileSizeKB = 100
    )

    Write-Host "📂 Analyzing codebase: $Path" -ForegroundColor Gray

    # Safety check: Don't analyze huge directories like home directory
    $safeDirectories = @(
        "C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot",
        "C:\Users\Owner\Documents\",
        "C:\Users\Owner\Desktop\"
    )

    $isSafe = $false
    foreach ($safeDir in $safeDirectories) {
        if ($Path -like "$safeDir*") {
            $isSafe = $true
            break
        }
    }

    if (-not $isSafe) {
        Write-Host "⚠️  Skipping large directory scan for safety" -ForegroundColor Yellow
        return "CODEBASE: $Path`nNOTE: Directory too large to scan. Navigate to a specific project folder for detailed analysis."
    }

    # Detect language (simplified to avoid scanning)
    $language = "Unknown"
    Write-Host "🔍 Language: $language" -ForegroundColor Gray

    # Get file extensions based on language
    $extensions = switch ($language) {
        'Python' { @('*.py') }
        'JavaScript' { @('*.js', '*.jsx') }
        'TypeScript' { @('*.ts', '*.tsx') }
        'CSharp' { @('*.cs') }
        'Java' { @('*.java') }
        'Go' { @('*.go') }
        'Rust' { @('*.rs') }
        'CPP' { @('*.cpp', '*.hpp', '*.h', '*.c') }
        default { @('*.py', '*.js', '*.ts', '*.cs', '*.java', '*.go', '*.rs', '*.cpp') }
    }

    # Build context
    $context = "CODEBASE: $Path`n"
    $context += "LANGUAGE: $language`n"
    $context += "TIMESTAMP: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n`n"

    # Get directory tree
    $context += "DIRECTORY STRUCTURE:`n"
    try {
        $dirs = Get-ChildItem -Path $Path -Directory -Recurse -ErrorAction SilentlyContinue | Select-Object -First 20
        foreach ($dir in $dirs) {
            $relativePath = $dir.FullName.Replace($Path, '').TrimStart('\')
            $context += "  📁 $relativePath`n"
        }
    } catch {
        Write-MambaLog -Level 'WARN' -Message "Failed to read directory structure: $_"
    }

    $context += "`nKEY FILES:`n"

    # Get code files
    $files = @()
    foreach ($ext in $extensions) {
        $files += Get-ChildItem -Path $Path -Filter $ext -Recurse -File -ErrorAction SilentlyContinue |
                  Where-Object { $_.Length -lt ($MaxFileSizeKB * 1KB) } |
                  Select-Object -First $MaxFiles
    }

    $fileCount = 0
    foreach ($file in $files) {
        if ($fileCount -ge $MaxFiles) { break }

        $relativePath = $file.FullName.Replace($Path, '').TrimStart('\')
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue

        if ($content) {
            $extension = $file.Extension.TrimStart('.')
            $context += "`nFILE: $relativePath`n"
            $context += "``````$extension`n$content`n``````n"
            $fileCount++
        }
    }

    $context += "`nTotal files analyzed: $fileCount`n"

    return $context
}

# ══════════════════════════════════════════════════════════════════
# TOOL 1: FILE SYSTEM TOOL
# ══════════════════════════════════════════════════════════════════

function Read-CodeFile {
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath
    )

    if (-not (Test-SafePath -Path $FilePath)) {
        return "❌ Access denied to: $FilePath"
    }

    if (-not (Test-Path $FilePath)) {
        return "❌ File not found: $FilePath"
    }

    try {
        $content = Get-Content $FilePath -Raw
        Write-MambaLog -Level 'INFO' -Message "Read file" -Data @{path=$FilePath; size=$content.Length}
        return $content
    } catch {
        Write-MambaLog -Level 'ERROR' -Message "Failed to read file: $_" -Data @{path=$FilePath}
        return "❌ Error reading file: $_"
    }
}

function Write-CodeFile {
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath,
        [Parameter(Mandatory=$true)]
        [string]$Content
    )

    if (-not (Test-SafeOperation -Operation "Write-CodeFile" -Path $FilePath)) {
        return "❌ Operation blocked for: $FilePath"
    }

    try {
        # Create backup if file exists
        if (Test-Path $FilePath) {
            $backupPath = "$FilePath.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
            Copy-Item $FilePath $backupPath
            Write-Host "📦 Backup created: $backupPath" -ForegroundColor Gray
        }

        Set-Content -Path $FilePath -Value $Content
        Write-MambaLog -Level 'INFO' -Message "Wrote file" -Data @{path=$FilePath; size=$Content.Length}
        return "✅ File written successfully: $FilePath"
    } catch {
        Write-MambaLog -Level 'ERROR' -Message "Failed to write file: $_" -Data @{path=$FilePath}
        return "❌ Error writing file: $_"
    }
}

function Search-Codebase {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Pattern,
        [string]$Path = (Get-CurrentCodebase),
        [string[]]$Extensions = @('*.py', '*.js', '*.ts', '*.cs', '*.java')
    )

    if (-not (Test-SafePath -Path $Path)) {
        return "❌ Access denied to: $Path"
    }

    Write-Host "🔍 Searching for: $Pattern" -ForegroundColor Yellow

    $results = @()
    foreach ($ext in $Extensions) {
        $files = Get-ChildItem -Path $Path -Filter $ext -Recurse -File -ErrorAction SilentlyContinue

        foreach ($file in $files) {
            $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
            if ($content -match $Pattern) {
                $relativePath = $file.FullName.Replace($Path, '').TrimStart('\')
                $results += "📄 $relativePath"

                # Find matching lines
                $lines = $content -split "`n"
                for ($i = 0; $i -lt $lines.Count; $i++) {
                    if ($lines[$i] -match $Pattern) {
                        $results += "   Line $($i+1): $($lines[$i].Trim())"
                    }
                }
            }
        }
    }

    if ($results.Count -eq 0) {
        return "No matches found for: $Pattern"
    }

    Write-MambaLog -Level 'INFO' -Message "Search completed" -Data @{pattern=$Pattern; matches=$results.Count}
    return $results -join "`n"
}

# ══════════════════════════════════════════════════════════════════
# TOOL 2: GITHUB TOOL
# ══════════════════════════════════════════════════════════════════

function Search-GitHubRepos {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Query,
        [string]$Language,
        [int]$MinStars = 100
    )

    Write-Host "🔍 Searching GitHub for: $Query" -ForegroundColor Yellow

    $headers = @{
        'Accept' = 'application/vnd.github+json'
        'User-Agent' = 'Mamba-AI-PowerShell'
    }

    $searchQuery = $Query
    if ($Language) { $searchQuery += " language:$Language" }
    $searchQuery += " stars:>$MinStars"

    try {
        $encodedQuery = [System.Web.HttpUtility]::UrlEncode($searchQuery)
        $url = "https://api.github.com/search/repositories?q=$encodedQuery&sort=stars&order=desc&per_page=10"

        $response = Invoke-RestMethod -Uri $url -Headers $headers -ErrorAction Stop

        $results = @()
        foreach ($repo in $response.items) {
            $results += @{
                Name = $repo.full_name
                Stars = $repo.stargazers_count
                Description = $repo.description
                Language = $repo.language
                URL = $repo.html_url
            }
        }

        Write-MambaLog -Level 'INFO' -Message "GitHub search completed" -Data @{query=$Query; results=$results.Count}
        return $results

    } catch {
        Write-MambaLog -Level 'ERROR' -Message "GitHub search failed: $_"
        return @{Error = "Failed to search GitHub: $_"}
    }
}

function Get-GitHubFileContent {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Owner,
        [Parameter(Mandatory=$true)]
        [string]$Repo,
        [Parameter(Mandatory=$true)]
        [string]$Path
    )

    $branches = @('main', 'master')

    foreach ($branch in $branches) {
        try {
            $url = "https://raw.githubusercontent.com/$Owner/$Repo/$branch/$Path"
            $content = Invoke-RestMethod -Uri $url -ErrorAction Stop
            Write-MambaLog -Level 'INFO' -Message "Retrieved GitHub file" -Data @{repo="$Owner/$Repo"; path=$Path}
            return $content
        } catch {
            continue
        }
    }

    Write-MambaLog -Level 'WARN' -Message "File not found on GitHub" -Data @{repo="$Owner/$Repo"; path=$Path}
    return "❌ File not found: $Path in $Owner/$Repo"
}

# ══════════════════════════════════════════════════════════════════
# TOOL 3: WEB TOOL (Stack Overflow + Documentation)
# ══════════════════════════════════════════════════════════════════

function Search-StackOverflow {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Query,
        [string]$Tag
    )

    Write-Host "🌐 Searching Stack Overflow for: $Query" -ForegroundColor Yellow

    try {
        Add-Type -AssemblyName System.Web

        $encodedQuery = [System.Web.HttpUtility]::UrlEncode($Query)
        $url = "https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&q=$encodedQuery&site=stackoverflow"

        if ($Tag) {
            $url += "&tagged=$Tag"
        }

        $response = Invoke-RestMethod -Uri $url -ErrorAction Stop

        $results = @()
        foreach ($item in $response.items | Select-Object -First 5) {
            $results += @{
                Title = $item.title
                Link = $item.link
                Score = $item.score
                Answered = $item.is_answered
                AnswerCount = $item.answer_count
                QuestionId = $item.question_id
            }
        }

        Write-MambaLog -Level 'INFO' -Message "Stack Overflow search completed" -Data @{query=$Query; results=$results.Count}
        return $results

    } catch {
        Write-MambaLog -Level 'ERROR' -Message "Stack Overflow search failed: $_"
        return @{Error = "Failed to search Stack Overflow: $_"}
    }
}

function Get-StackOverflowAnswer {
    param([int]$QuestionId)

    try {
        $url = "https://api.stackexchange.com/2.3/questions/$QuestionId/answers?order=desc&sort=votes&site=stackoverflow&filter=withbody"
        $response = Invoke-RestMethod -Uri $url -ErrorAction Stop

        if ($response.items.Count -gt 0) {
            return $response.items[0].body_markdown
        }

        return "No answers found"

    } catch {
        Write-MambaLog -Level 'ERROR' -Message "Failed to get Stack Overflow answer: $_"
        return "❌ Error: $_"
    }
}

function Scrape-Documentation {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Url
    )

    Write-Host "📖 Fetching documentation from: $Url" -ForegroundColor Yellow

    try {
        $html = Invoke-WebRequest -Uri $Url -ErrorAction Stop
        $textContent = $html.ParsedHtml.body.innerText

        Write-MambaLog -Level 'INFO' -Message "Documentation fetched" -Data @{url=$Url; length=$textContent.Length}
        return $textContent

    } catch {
        Write-MambaLog -Level 'ERROR' -Message "Failed to scrape documentation: $_"
        return "❌ Failed to fetch documentation: $_"
    }
}

# ══════════════════════════════════════════════════════════════════
# TOOL 4: YOUTUBE TOOL
# ══════════════════════════════════════════════════════════════════

function Search-YouTubeTutorials {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Query,
        [int]$MaxResults = 5
    )

    if (-not $script:YOUTUBE_API_KEY) {
        Write-Host "⚠️  YouTube API key not set. Set `$env:YOUTUBE_API_KEY" -ForegroundColor Yellow
        return @{Error = "YouTube API key not configured"}
    }

    Write-Host "🎥 Searching YouTube for: $Query" -ForegroundColor Yellow

    try {
        Add-Type -AssemblyName System.Web
        $encodedQuery = [System.Web.HttpUtility]::UrlEncode("$Query tutorial")
        $url = "https://www.googleapis.com/youtube/v3/search?part=snippet&q=$encodedQuery&type=video&maxResults=$MaxResults&key=$script:YOUTUBE_API_KEY"

        $response = Invoke-RestMethod -Uri $url -ErrorAction Stop

        $results = @()
        foreach ($item in $response.items) {
            $results += @{
                Title = $item.snippet.title
                VideoId = $item.id.videoId
                Link = "https://www.youtube.com/watch?v=$($item.id.videoId)"
                Description = $item.snippet.description
            }
        }

        Write-MambaLog -Level 'INFO' -Message "YouTube search completed" -Data @{query=$Query; results=$results.Count}
        return $results

    } catch {
        Write-MambaLog -Level 'ERROR' -Message "YouTube search failed: $_"
        return @{Error = "Failed to search YouTube: $_"}
    }
}

# ══════════════════════════════════════════════════════════════════
# TOOL 5: CODE ANALYSIS TOOL
# ══════════════════════════════════════════════════════════════════

function Analyze-PythonCode {
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath
    )

    if (-not (Test-Path $FilePath)) {
        return @{Error = "File not found: $FilePath"}
    }

    Write-Host "🔍 Analyzing Python code: $FilePath" -ForegroundColor Yellow

    $analysis = @{
        FilePath = $FilePath
        Language = 'Python'
        SyntaxValid = $false
        Errors = @()
        Warnings = @()
        Imports = @()
        Functions = @()
        Classes = @()
        LineCount = 0
    }

    try {
        # Get file content
        $content = Get-Content $FilePath -Raw
        $analysis.LineCount = ($content -split "`n").Count

        # Check syntax using Python's compile
        $syntaxCheck = python -c "import sys; compile(open('$FilePath').read(), '$FilePath', 'exec'); print('VALID')" 2>&1
        $analysis.SyntaxValid = ($syntaxCheck -match 'VALID')

        if (-not $analysis.SyntaxValid) {
            $analysis.Errors += "Syntax error: $syntaxCheck"
        }

        # Extract imports, functions, classes using AST
        $astScript = @"
import ast
import json
try:
    with open('$FilePath', 'r') as f:
        tree = ast.parse(f.read())

    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend([alias.name for alias in node.names])
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)

    print(json.dumps({'functions': functions, 'classes': classes, 'imports': imports}))
except Exception as e:
    print(json.dumps({'error': str(e)}))
"@

        $astResult = python -c $astScript 2>&1 | ConvertFrom-Json
        if ($astResult.error) {
            $analysis.Errors += "AST parsing failed: $($astResult.error)"
        } else {
            $analysis.Functions = $astResult.functions
            $analysis.Classes = $astResult.classes
            $analysis.Imports = $astResult.imports
        }

        Write-MambaLog -Level 'INFO' -Message "Python code analysis completed" -Data @{file=$FilePath; valid=$analysis.SyntaxValid}

    } catch {
        Write-MambaLog -Level 'ERROR' -Message "Python analysis failed: $_"
        $analysis.Errors += "Analysis failed: $_"
    }

    return $analysis
}

function Analyze-JavaScriptCode {
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath
    )

    if (-not (Test-Path $FilePath)) {
        return @{Error = "File not found: $FilePath"}
    }

    Write-Host "🔍 Analyzing JavaScript code: $FilePath" -ForegroundColor Yellow

    $analysis = @{
        FilePath = $FilePath
        Language = 'JavaScript'
        SyntaxValid = $false
        Errors = @()
        Warnings = @()
        LineCount = 0
    }

    try {
        $content = Get-Content $FilePath -Raw
        $analysis.LineCount = ($content -split "`n").Count

        # Check syntax using Node.js --check
        $syntaxCheck = node --check $FilePath 2>&1
        $analysis.SyntaxValid = ($LASTEXITCODE -eq 0)

        if (-not $analysis.SyntaxValid) {
            $analysis.Errors += "Syntax error: $syntaxCheck"
        }

        Write-MambaLog -Level 'INFO' -Message "JavaScript code analysis completed" -Data @{file=$FilePath; valid=$analysis.SyntaxValid}

    } catch {
        Write-MambaLog -Level 'WARN' -Message "JavaScript analysis failed: $_"
        $analysis.Errors += "Analysis failed: $_"
    }

    return $analysis
}

function Analyze-CodeFile {
    <#
    .SYNOPSIS
    Universal code analyzer - detects language and runs appropriate analysis
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath
    )

    if (-not (Test-Path $FilePath)) {
        Write-Host "❌ File not found: $FilePath" -ForegroundColor Red
        return @{Error = "File not found"}
    }

    $extension = [System.IO.Path]::GetExtension($FilePath).ToLower()

    $analysis = switch ($extension) {
        '.py' { Analyze-PythonCode -FilePath $FilePath }
        '.js' { Analyze-JavaScriptCode -FilePath $FilePath }
        '.jsx' { Analyze-JavaScriptCode -FilePath $FilePath }
        '.ts' { Analyze-JavaScriptCode -FilePath $FilePath }
        '.tsx' { Analyze-JavaScriptCode -FilePath $FilePath }
        default {
            @{
                FilePath = $FilePath
                Language = 'Unknown'
                Message = "Unsupported file type: $extension"
            }
        }
    }

    # Display results
    Write-Host "`n📊 Analysis Results:" -ForegroundColor Yellow
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
    Write-Host "File: $($analysis.FilePath)" -ForegroundColor Cyan
    Write-Host "Language: $($analysis.Language)" -ForegroundColor Cyan
    Write-Host "Lines: $($analysis.LineCount)" -ForegroundColor Gray

    if ($analysis.SyntaxValid -ne $null) {
        $statusColor = if ($analysis.SyntaxValid) { 'Green' } else { 'Red' }
        $statusIcon = if ($analysis.SyntaxValid) { '✅' } else { '❌' }
        Write-Host "$statusIcon Syntax: " -NoNewline -ForegroundColor $statusColor
        Write-Host $(if ($analysis.SyntaxValid) { "VALID" } else { "INVALID" }) -ForegroundColor $statusColor
    }

    if ($analysis.Functions) {
        Write-Host "`nFunctions ($($analysis.Functions.Count)):" -ForegroundColor Yellow
        $analysis.Functions | ForEach-Object { Write-Host "  - $_" -ForegroundColor Gray }
    }

    if ($analysis.Classes) {
        Write-Host "`nClasses ($($analysis.Classes.Count)):" -ForegroundColor Yellow
        $analysis.Classes | ForEach-Object { Write-Host "  - $_" -ForegroundColor Gray }
    }

    if ($analysis.Imports) {
        Write-Host "`nImports ($($analysis.Imports.Count)):" -ForegroundColor Yellow
        $analysis.Imports | Select-Object -Unique | ForEach-Object { Write-Host "  - $_" -ForegroundColor Gray }
    }

    if ($analysis.Errors.Count -gt 0) {
        Write-Host "`n❌ Errors:" -ForegroundColor Red
        $analysis.Errors | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    }

    if ($analysis.Warnings.Count -gt 0) {
        Write-Host "`n⚠️  Warnings:" -ForegroundColor Yellow
        $analysis.Warnings | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
    }

    return $analysis
}

function Auto-Fix-Code {
    <#
    .SYNOPSIS
    Automatically fix common code issues (formatting, imports)
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath
    )

    if (-not (Test-SafePath -Path $FilePath)) {
        Write-Host "❌ Access denied: $FilePath" -ForegroundColor Red
        return
    }

    if (-not (Test-Path $FilePath)) {
        Write-Host "❌ File not found: $FilePath" -ForegroundColor Red
        return
    }

    $extension = [System.IO.Path]::GetExtension($FilePath).ToLower()

    Write-Host "🔧 Auto-fixing code: $FilePath" -ForegroundColor Yellow

    try {
        switch ($extension) {
            '.py' {
                # Backup original
                $backupPath = "$FilePath.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
                Copy-Item $FilePath $backupPath
                Write-Host "📦 Backup created: $backupPath" -ForegroundColor Gray

                # Run autopep8 for formatting
                $result = python -m autopep8 --in-place --aggressive --aggressive $FilePath 2>&1
                Write-Host "✅ Applied autopep8 formatting" -ForegroundColor Green

                # Run isort for import sorting
                $result = python -m isort $FilePath 2>&1
                Write-Host "✅ Sorted imports with isort" -ForegroundColor Green

                Write-Host "`n🎉 Python code auto-fixed successfully!" -ForegroundColor Green
            }
            '.js' {
                Write-Host "⚠️  JavaScript auto-fix requires npm packages (prettier, eslint)" -ForegroundColor Yellow
                Write-Host "   Install with: npm install -g prettier eslint" -ForegroundColor Gray
            }
            default {
                Write-Host "⚠️  Auto-fix not available for: $extension" -ForegroundColor Yellow
            }
        }

        Write-MambaLog -Level 'INFO' -Message "Code auto-fix completed" -Data @{file=$FilePath}

    } catch {
        Write-Host "❌ Auto-fix failed: $_" -ForegroundColor Red
        Write-MambaLog -Level 'ERROR' -Message "Auto-fix failed: $_" -Data @{file=$FilePath}
    }
}

# ══════════════════════════════════════════════════════════════════
# AI BRAIN - GPT-4 / CLAUDE INTEGRATION
# ══════════════════════════════════════════════════════════════════

function Invoke-MambaAI {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Prompt,

        [string]$Model = $script:DEFAULT_MODEL,

        [switch]$IncludeCodebase,

        [string]$SystemPrompt = "You are Mamba AI v2, a genius-level coding assistant with access to powerful tools. You work with ANY codebase in ANY language. You can search GitHub, Stack Overflow, read files, analyze code, and research topics. Be concise, accurate, and provide working solutions with full explanations."
    )

    Write-Host "`n🐍 Mamba AI thinking..." -ForegroundColor Yellow

    try {
        # Add codebase context if requested
        $fullPrompt = $Prompt
        if ($IncludeCodebase) {
            Write-Host "📂 Loading codebase context..." -ForegroundColor Gray
            $codeContext = Get-CodebaseContext

            # Ensure context is a string
            if ($null -ne $codeContext) {
                $codeContextStr = [string]$codeContext
                $fullPrompt = "CODEBASE CONTEXT:`n" + $codeContextStr + "`n`nUSER QUESTION: " + $Prompt
            }
        }

        # Add to conversation history
        Add-ToHistory -Role 'user' -Content ([string]$fullPrompt)

        if ($Model -like "gpt-*") {
            $response = Invoke-OpenAI -Prompt $fullPrompt -SystemPrompt $SystemPrompt
        } elseif ($Model -like "claude-*") {
            $response = Invoke-Claude -Prompt $fullPrompt -SystemPrompt $SystemPrompt
        } else {
            throw "Unknown model: $Model"
        }

        # Add AI response to history
        Add-ToHistory -Role 'assistant' -Content $response

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
        Write-MambaLog -Level 'ERROR' -Message "AI request failed: $_"
        return $null
    }
}

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
        ) + $script:CONVERSATION_HISTORY + @(
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

    # Build messages array properly
    $messages = @()

    # Add conversation history (filter out timestamp field for API)
    foreach ($msg in $script:CONVERSATION_HISTORY) {
        if ($msg.role -and $msg.content) {
            $messages += @{
                role = $msg.role
                content = $msg.content
            }
        }
    }

    # Add current user message
    $messages += @{
        role = "user"
        content = $Prompt
    }

    $body = @{
        model = "claude-3-opus-20240229"
        max_tokens = 2000
        system = $SystemPrompt
        messages = $messages
    } | ConvertTo-Json -Depth 10

    $response = Invoke-RestMethod -Uri "https://api.anthropic.com/v1/messages" `
                                   -Method Post `
                                   -Headers $headers `
                                   -Body $body

    return $response.content[0].text
}

# ══════════════════════════════════════════════════════════════════
# HIGH-LEVEL AI COMMANDS
# ══════════════════════════════════════════════════════════════════

function ai {
    <#
    .SYNOPSIS
    Ask Mamba AI anything - works with ANY codebase

    .EXAMPLE
    ai "explain this code"
    ai "find all functions that handle errors"
    ai "how do I implement authentication?"
    #>
    param(
        [Parameter(ValueFromRemainingArguments=$true)]
        [string[]]$Question
    )

    $query = $Question -join " "

    if (-not $query) {
        Show-AIHelp
        return
    }

    # Only include codebase context for specific questions AND when in a safe directory
    $includeContext = $query -match "codebase|project|code|file|function|class|this project|what is this"

    $currentPath = Get-CurrentCodebase
    $isKobeBot = $currentPath -like "*kobe81_traderbot*"

    if ($includeContext -and $isKobeBot) {
        Invoke-MambaAI -Prompt $query -IncludeCodebase
    } else {
        # If not in trading bot folder, give a helpful message
        if ($includeContext -and -not $isKobeBot) {
            Write-Host ""
            Write-Host "💡 Tip: Navigate to your trading bot folder for code analysis:" -ForegroundColor Yellow
            Write-Host "   cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot" -ForegroundColor Cyan
            Write-Host ""
        }
        Invoke-MambaAI -Prompt $query
    }
}

function ai-code {
    <#
    .SYNOPSIS
    Generate production-ready code for a task
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Task
    )

    $prompt = "Write production-ready code for: $Task. Include error handling, comments, and best practices."
    Invoke-MambaAI -Prompt $prompt
}

function ai-fix {
    <#
    .SYNOPSIS
    Analyze and fix bugs in a file
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath
    )

    if (-not (Test-Path $FilePath)) {
        Write-Host "❌ File not found: $FilePath" -ForegroundColor Red
        return
    }

    $code = Read-CodeFile -FilePath $FilePath
    $prompt = "Fix any bugs or issues in this code. Explain what was wrong and provide the corrected version:`n`n``````n$code`n``````"

    Invoke-MambaAI -Prompt $prompt
}

function ai-review {
    <#
    .SYNOPSIS
    Code review like a senior developer
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath
    )

    if (-not (Test-Path $FilePath)) {
        Write-Host "❌ File not found: $FilePath" -ForegroundColor Red
        return
    }

    $code = Read-CodeFile -FilePath $FilePath
    $prompt = "Review this code like a senior developer. Check for bugs, performance issues, best practices, security. Provide specific improvements:`n`n``````n$code`n``````"

    Invoke-MambaAI -Prompt $prompt
}

function ai-analyze {
    <#
    .SYNOPSIS
    Comprehensive code analysis with AI insights
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath
    )

    if (-not (Test-Path $FilePath)) {
        Write-Host "❌ File not found: $FilePath" -ForegroundColor Red
        return
    }

    # Run code analysis
    $analysis = Analyze-CodeFile -FilePath $FilePath

    # Get AI insights
    $analysisJson = $analysis | ConvertTo-Json -Depth 10
    $code = Read-CodeFile -FilePath $FilePath

    $prompt = @"
Provide expert analysis and recommendations for this code:

CODE ANALYSIS RESULTS:
$analysisJson

FULL CODE:
``````
$code
``````

Please provide:
1. Code quality assessment (1-10)
2. Key strengths
3. Areas for improvement
4. Security concerns
5. Performance optimization suggestions
6. Best practices compliance
"@

    Invoke-MambaAI -Prompt $prompt
}

function ai-research {
    <#
    .SYNOPSIS
    Deep research across GitHub + Stack Overflow + Documentation
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Topic
    )

    Write-Host "`n🔬 Starting deep research on: $Topic" -ForegroundColor Yellow
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

    # Search GitHub
    Write-Host "`n📦 Searching GitHub repositories..." -ForegroundColor Cyan
    $githubResults = Search-GitHubRepos -Query $Topic -MinStars 100

    # Search Stack Overflow
    Write-Host "`n💬 Searching Stack Overflow..." -ForegroundColor Cyan
    $soResults = Search-StackOverflow -Query $Topic

    # Compile research
    $researchData = "`n=== GITHUB REPOSITORIES ===`n"
    foreach ($repo in $githubResults) {
        $researchData += "`n$($repo.Name) ⭐ $($repo.Stars)`n"
        $researchData += "$($repo.Description)`n"
        $researchData += "$($repo.URL)`n"
    }

    $researchData += "`n`n=== STACK OVERFLOW SOLUTIONS ===`n"
    foreach ($question in $soResults) {
        $researchData += "`n$($question.Title) (Score: $($question.Score))`n"
        $researchData += "$($question.Link)`n"
    }

    # Synthesize with AI
    Write-Host "`n🧠 Synthesizing research findings..." -ForegroundColor Yellow
    $prompt = "Based on this research data, provide a comprehensive summary of best practices, common patterns, and recommendations for: $Topic`n`nRESEARCH DATA:`n$researchData"

    Invoke-MambaAI -Prompt $prompt
}

function ai-debug {
    <#
    .SYNOPSIS
    Debug error messages with Stack Overflow solutions
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Error
    )

    Write-Host "`n🔍 Analyzing error..." -ForegroundColor Yellow

    # Search Stack Overflow
    $soResults = Search-StackOverflow -Query $Error

    $researchData = "ERROR: $Error`n`nSTACK OVERFLOW SOLUTIONS:`n"
    foreach ($question in $soResults | Select-Object -First 3) {
        $researchData += "`n$($question.Title) (Score: $($question.Score))`n"
        $researchData += "$($question.Link)`n"

        if ($question.Answered) {
            $answer = Get-StackOverflowAnswer -QuestionId $question.QuestionId
            $researchData += "TOP ANSWER:`n$answer`n"
        }
    }

    # Synthesize solution
    $prompt = "Based on these Stack Overflow solutions, explain the root cause and provide a step-by-step fix:`n`n$researchData"

    Invoke-MambaAI -Prompt $prompt
}

function ai-github {
    <#
    .SYNOPSIS
    Search GitHub repos for examples and best practices
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Query,
        [string]$Language
    )

    $repos = Search-GitHubRepos -Query $Query -Language $Language

    Write-Host "`n📦 Top GitHub Repositories:" -ForegroundColor Yellow
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

    foreach ($repo in $repos) {
        Write-Host "`n⭐ $($repo.Stars) - $($repo.Name)" -ForegroundColor Cyan
        Write-Host "   $($repo.Description)" -ForegroundColor Gray
        Write-Host "   $($repo.URL)" -ForegroundColor DarkCyan
    }
}

function ai-youtube {
    <#
    .SYNOPSIS
    Find YouTube tutorials on a topic
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Topic
    )

    $videos = Search-YouTubeTutorials -Query $Topic

    if ($videos.Error) {
        Write-Host $videos.Error -ForegroundColor Red
        return
    }

    Write-Host "`n🎥 YouTube Tutorials:" -ForegroundColor Yellow
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

    foreach ($video in $videos) {
        Write-Host "`n📺 $($video.Title)" -ForegroundColor Cyan
        Write-Host "   $($video.Link)" -ForegroundColor DarkCyan
        Write-Host "   $($video.Description.Substring(0, [Math]::Min(100, $video.Description.Length)))..." -ForegroundColor Gray
    }
}

function ai-chat {
    <#
    .SYNOPSIS
    Interactive chat mode with conversation history
    #>
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║       🐍 MAMBA AI V2 - INTERACTIVE GENIUS MODE 🐍               ║" -ForegroundColor Yellow
    Write-Host "║          Works with ANY codebase, ANY language                   ║" -ForegroundColor Yellow
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Current folder: $(Get-CurrentCodebase)" -ForegroundColor Gray
    Write-Host "Detected language: $(Get-CodebaseLanguage -Path (Get-CurrentCodebase))" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Type your questions. Type 'exit' to quit." -ForegroundColor Gray
    Write-Host ""

    while ($true) {
        Write-Host "You: " -NoNewline -ForegroundColor Green
        $input = Read-Host

        if ($input -eq "exit" -or $input -eq "quit") {
            Write-Host "👋 Goodbye! Conversation history saved." -ForegroundColor Yellow
            break
        }

        if ($input -eq "clear") {
            $script:CONVERSATION_HISTORY = @()
            Save-ConversationHistory -History @()
            Write-Host "🧹 Conversation history cleared" -ForegroundColor Yellow
            continue
        }

        if ($input) {
            Invoke-MambaAI -Prompt $input -IncludeCodebase
        }
    }
}

function ai-history {
    <#
    .SYNOPSIS
    Show conversation history
    #>
    $history = Load-ConversationHistory

    Write-Host "`n📜 Conversation History ($($history.Count) messages):" -ForegroundColor Yellow
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

    foreach ($msg in $history | Select-Object -Last 10) {
        $timestamp = [datetime]::Parse($msg.timestamp).ToString("HH:mm:ss")
        $color = if ($msg.role -eq 'user') { 'Green' } else { 'Cyan' }
        $prefix = if ($msg.role -eq 'user') { '👤' } else { '🐍' }

        Write-Host "`n[$timestamp] $prefix $($msg.role):" -ForegroundColor $color
        Write-Host $msg.content.Substring(0, [Math]::Min(200, $msg.content.Length)) -ForegroundColor Gray
        if ($msg.content.Length -gt 200) {
            Write-Host "... (truncated)" -ForegroundColor DarkGray
        }
    }
}

function Show-AIHelp {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║         🐍 MAMBA AI V2 - GENIUS-LEVEL ASSISTANT 🐍               ║" -ForegroundColor Yellow
    Write-Host "║           Works with ANY codebase, ANY language                  ║" -ForegroundColor Yellow
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Current Directory:" -ForegroundColor Cyan
    Write-Host "  $(Get-CurrentCodebase)" -ForegroundColor Gray
    Write-Host "  Language: $(Get-CodebaseLanguage -Path (Get-CurrentCodebase))" -ForegroundColor Gray
    Write-Host ""
    Write-Host "🤖 BASIC COMMANDS:" -ForegroundColor Cyan
    Write-Host "  ai [question]              " -NoNewline; Write-Host "Ask anything about current codebase" -ForegroundColor Yellow
    Write-Host "  ai-chat                    " -NoNewline; Write-Host "Interactive chat mode" -ForegroundColor Yellow
    Write-Host "  ai-history                 " -NoNewline; Write-Host "Show conversation history" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "💻 CODE COMMANDS:" -ForegroundColor Cyan
    Write-Host "  ai-code [task]             " -NoNewline; Write-Host "Generate code for a task" -ForegroundColor Yellow
    Write-Host "  ai-fix [file]              " -NoNewline; Write-Host "Fix bugs in a file" -ForegroundColor Yellow
    Write-Host "  ai-review [file]           " -NoNewline; Write-Host "Code review" -ForegroundColor Yellow
    Write-Host "  ai-analyze [file]          " -NoNewline; Write-Host "Deep code analysis with AI" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🔬 RESEARCH COMMANDS:" -ForegroundColor Cyan
    Write-Host "  ai-research [topic]        " -NoNewline; Write-Host "Deep research (GitHub + SO + Docs)" -ForegroundColor Yellow
    Write-Host "  ai-github [query]          " -NoNewline; Write-Host "Search GitHub repos" -ForegroundColor Yellow
    Write-Host "  ai-youtube [topic]         " -NoNewline; Write-Host "Find YouTube tutorials" -ForegroundColor Yellow
    Write-Host "  ai-debug [error]           " -NoNewline; Write-Host "Debug with SO solutions" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  ai explain this codebase" -ForegroundColor Gray
    Write-Host "  ai-research best practices for Python async" -ForegroundColor Gray
    Write-Host "  ai-github backtesting framework Python" -ForegroundColor Gray
    Write-Host "  ai-debug 'KeyError: close_price'" -ForegroundColor Gray
    Write-Host ""
}

# ══════════════════════════════════════════════════════════════════
# INITIALIZATION
# ══════════════════════════════════════════════════════════════════

Write-Host ""
Write-Host "🐍 Mamba AI v2 loaded! Type " -NoNewline -ForegroundColor Yellow
Write-Host "ai" -NoNewline -ForegroundColor Cyan
Write-Host " for help" -ForegroundColor Yellow
Write-Host "📂 Current: $(Get-CurrentCodebase)" -ForegroundColor Gray

# Show which AI provider is active
if ($script:DEFAULT_MODEL -like "claude-*") {
    Write-Host "🤖 AI: " -NoNewline -ForegroundColor Gray
    Write-Host "Claude (Anthropic)" -ForegroundColor Green
} elseif ($script:DEFAULT_MODEL -like "gpt-*") {
    Write-Host "🤖 AI: " -NoNewline -ForegroundColor Gray
    Write-Host "ChatGPT (OpenAI)" -ForegroundColor Green
}

Write-Host ""

# Functions are available when dot-sourced (no Export-ModuleMember needed)
