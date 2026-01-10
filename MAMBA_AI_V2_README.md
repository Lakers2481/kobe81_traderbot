# 🐍 MAMBA AI V2 - GENIUS-LEVEL UNIVERSAL ASSISTANT

## Overview

**Mamba AI v2** is a QUANT-LEVEL AI assistant that surpasses ChatGPT and Claude through powerful tool integration. It works with **ANY codebase** in **ANY language** and has direct access to GitHub, Stack Overflow, YouTube, and your file system.

### Key Features

| Feature | Description |
|---------|-------------|
| **Universal Codebase Support** | Works with Python, JavaScript, TypeScript, C#, Java, Go, Rust, C++ |
| **Language Auto-Detection** | Automatically detects your project's language |
| **Persistent Memory** | Conversation history survives PowerShell restarts |
| **Security Sandbox** | Protects Windows system directories from accidental damage |
| **Multi-Tool Integration** | GitHub, Stack Overflow, YouTube, File System, Code Analysis |
| **Quant-Level Accuracy** | Cross-references multiple sources for verification |

---

## 🚀 Quick Start

### Prerequisites

1. **API Keys** (at least one required):
   ```powershell
   $env:OPENAI_API_KEY = 'sk-your-key-here'        # For GPT-4
   $env:ANTHROPIC_API_KEY = 'sk-ant-your-key'      # For Claude (optional)
   $env:YOUTUBE_API_KEY = 'your-youtube-key'       # For YouTube (optional)
   ```

2. **Python** (for code analysis features):
   ```bash
   pip install autopep8 isort
   ```

3. **Node.js** (for JavaScript analysis, optional):
   ```bash
   npm install -g node
   ```

### Activation

1. **Restart PowerShell** - Mamba AI v2 loads automatically from your profile
2. **Verify Installation**:
   ```powershell
   ai
   ```
   You should see the help menu with all available commands.

---

## 💻 Basic Commands

### Ask Questions About Code

```powershell
# Ask anything about your current codebase
ai "explain what this code does"
ai "find all functions that handle errors"
ai "how does the authentication work?"
ai "what libraries are being used?"
```

**Key Feature**: Automatically includes your entire codebase context (up to 50 files).

### Interactive Chat Mode

```powershell
ai-chat
```

- Maintains conversation history across the session
- Type `clear` to reset history
- Type `exit` to quit
- History persists between PowerShell sessions

### View Conversation History

```powershell
ai-history
```

Shows last 10 messages with timestamps.

---

## 🛠️ Code Commands

### Generate Code

```powershell
ai-code "create a function to validate email addresses"
ai-code "write a class for managing database connections"
ai-code "implement a binary search algorithm"
```

**Output**: Production-ready code with error handling, comments, and best practices.

### Fix Bugs

```powershell
ai-fix C:\path\to\script.py
ai-fix .\mycode.js
```

**Process**:
1. Reads the file
2. Analyzes for bugs
3. Explains what's wrong
4. Provides corrected version

### Code Review

```powershell
ai-review C:\path\to\script.py
```

**Checks**:
- Bugs
- Performance issues
- Security vulnerabilities
- Best practices compliance
- Readability

### Deep Code Analysis

```powershell
ai-analyze C:\path\to\script.py
```

**Includes**:
- Syntax validation
- Import/function/class extraction
- Line count, complexity metrics
- AI-powered quality assessment (1-10 score)
- Security concerns
- Performance optimization suggestions

---

## 🔬 Research Commands

### Deep Research (GitHub + Stack Overflow + Docs)

```powershell
ai-research "best practices for Python async error handling"
ai-research "TypeScript vs JavaScript performance"
ai-research "how to implement JWT authentication"
```

**Process**:
1. Searches GitHub (top repos with 100+ stars)
2. Searches Stack Overflow (top 5 answered questions)
3. Scrapes official documentation (when available)
4. Synthesizes findings with AI

**Output**: Comprehensive summary with best practices, common patterns, and recommendations.

### Search GitHub Repos

```powershell
ai-github "backtesting framework Python"
ai-github "trading bot" -Language Python
```

**Returns**:
- Top 10 repos sorted by stars
- Repository descriptions
- Direct GitHub URLs

### Find YouTube Tutorials

```powershell
ai-youtube "Python asyncio tutorial"
ai-youtube "machine learning explained"
```

**Returns**:
- Top 5 tutorial videos
- Direct YouTube links
- Video descriptions

### Debug Errors

```powershell
ai-debug "KeyError: 'close_price'"
ai-debug "TypeError: unsupported operand type(s) for +: 'int' and 'str'"
```

**Process**:
1. Searches Stack Overflow for the error
2. Retrieves top-voted answers
3. Synthesizes solution with AI
4. Provides root cause explanation + step-by-step fix

---

## 📂 File System Commands

### Read Files

```powershell
$content = Read-CodeFile -FilePath "C:\path\to\file.py"
```

**Features**:
- Security sandbox (blocks system directories)
- Encoding auto-detection
- Logging

### Write Files

```powershell
Write-CodeFile -FilePath "C:\path\to\file.py" -Content $newCode
```

**Safety**:
- Creates automatic backup (`.backup_YYYYMMDD_HHmmss`)
- Requires confirmation for destructive operations
- Logs all writes

### Search Codebase

```powershell
Search-Codebase -Pattern "def.*authenticate" -Extensions @('*.py', '*.js')
```

**Returns**:
- All matching files
- Line numbers
- Matching lines of code

---

## 🔍 Code Analysis Tools

### Analyze Code File

```powershell
Analyze-CodeFile -FilePath "C:\path\to\script.py"
```

**Detects**:
- Language (Python, JavaScript, TypeScript)
- Syntax validity
- Functions and classes
- Imports/dependencies
- Line count
- Errors and warnings

**Example Output**:
```
📊 Analysis Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: C:\trading\strategy.py
Language: Python
Lines: 250
✅ Syntax: VALID

Functions (12):
  - calculate_rsi
  - generate_signals
  - backtest_strategy
  ...

Classes (3):
  - TradingStrategy
  - PortfolioManager
  - RiskCalculator

Imports (15):
  - pandas
  - numpy
  - ta
  ...
```

### Auto-Fix Code

```powershell
Auto-Fix-Code -FilePath "C:\path\to\script.py"
```

**Python Auto-Fix**:
1. Creates backup
2. Runs `autopep8` (formatting)
3. Runs `isort` (import sorting)

**Requires**:
```bash
pip install autopep8 isort
```

---

## 🧠 How It Works

### Universal Codebase Detection

When you ask a question, Mamba AI v2:

1. **Detects Current Directory**: Uses `$PWD` (wherever you navigated in PowerShell)
2. **Identifies Language**: Scans for Python, JavaScript, TypeScript, C#, Java, Go, Rust, C++ indicators
3. **Builds Context**: Reads up to 50 files (max 100KB each)
4. **Includes Structure**: Directory tree + key files
5. **Sends to AI**: GPT-4 or Claude with full context

**Example**:
```powershell
cd C:\Users\Owner\Documents\MyPythonProject
ai "explain the architecture"  # Works with MyPythonProject

cd C:\Users\Owner\Desktop\JavaApp
ai "find security vulnerabilities"  # Works with JavaApp
```

### Security Sandbox

**Blocked Paths** (cannot read/write):
- `C:\Windows\System32`
- `C:\Program Files\WindowsPowerShell`
- Registry paths
- Windows startup locations

**Allowed Paths**:
- `C:\Users\Owner\*` (your user directory)
- `C:\Temp`

**Dangerous Commands** (blocked):
- `Remove-Item -Recurse -Force`
- `Format-Volume`
- `rm -rf`
- `del /F /S /Q`

**Confirmation Required**:
- `Remove-Item`
- `Move-Item`
- `Rename-Item`

### Conversation Persistence

**Storage Location**: `C:\Users\Owner\.mamba\conversation_history.json`

**Features**:
- Survives PowerShell restarts
- Keeps last 100 messages
- Timestamped entries
- Role-based (user/assistant)

**Commands**:
```powershell
ai-history           # View last 10 messages
ai-chat → clear      # Clear history manually
```

---

## 📊 Configuration

### State Directory

All Mamba AI v2 data is stored in:
```
C:\Users\Owner\.mamba\
├── conversation_history.json    # Persistent chat history
├── mamba.log                    # Activity logs
└── config.json                  # Configuration (future use)
```

### Logging

All operations are logged with:
- Timestamp
- Log level (INFO, WARN, ERROR)
- Message
- Metadata (file paths, API calls, etc.)

**View Logs**:
```powershell
Get-Content C:\Users\Owner\.mamba\mamba.log -Tail 50
```

---

## 🎯 Use Cases

### 1. Learning a New Codebase

```powershell
cd C:\Projects\UnknownRepo
ai "give me a high-level overview of this codebase"
ai "what does the main entry point do?"
ai "show me all the API endpoints"
```

### 2. Debugging Production Issues

```powershell
ai-debug "ConnectionError: [Errno 111] Connection refused"
# Gets Stack Overflow solutions + AI synthesis
```

### 3. Code Quality Improvement

```powershell
ai-analyze .\mycode.py
# Gets syntax validation, metrics, AI quality assessment

Auto-Fix-Code .\mycode.py
# Applies formatting, import sorting
```

### 4. Research Before Implementation

```powershell
ai-research "how to implement rate limiting in Python"
# Gets GitHub examples + Stack Overflow discussions + AI summary

ai-github "rate limiting Python"
# Direct links to top repos
```

### 5. Code Review Before Commit

```powershell
ai-review .\new_feature.py
# Senior developer-level review
```

### 6. Learning From Videos

```powershell
ai-youtube "Python decorators explained"
# Top 5 tutorial videos with links
```

---

## 🚨 Troubleshooting

### Issue: "API key not set"

**Solution**:
```powershell
$env:OPENAI_API_KEY = 'sk-your-key-here'
# Then restart PowerShell or reload profile
reload
```

### Issue: Python modules not found (autopep8, isort)

**Solution**:
```bash
pip install autopep8 isort
```

### Issue: "Access denied" errors

**Cause**: Trying to access protected Windows directories

**Solution**: Navigate to your user directory
```powershell
cd $env:USERPROFILE
cd C:\Users\Owner\Documents
```

### Issue: Conversation history not persisting

**Check**:
```powershell
Test-Path C:\Users\Owner\.mamba\conversation_history.json
# Should return True
```

**Fix**:
```powershell
New-Item -ItemType Directory -Path C:\Users\Owner\.mamba -Force
```

### Issue: "Node.js not found" for JavaScript analysis

**Solution**: Install Node.js from https://nodejs.org/

---

## 📝 Examples

### Example 1: Analyze Trading Bot

```powershell
cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot
ai "explain the overall architecture of this trading system"
ai "show me the risk management implementation"
ai-analyze .\strategies\dual_strategy.py
```

### Example 2: Learn React Project

```powershell
cd C:\Projects\ReactApp
ai "what components are in this project?"
ai "how does state management work here?"
ai-research "React hooks best practices"
```

### Example 3: Debug API Issue

```powershell
ai-debug "HTTPError: 429 Too Many Requests"
ai-research "API rate limiting strategies"
ai-github "rate limiter Python"
```

### Example 4: Code Review Workflow

```powershell
# Before committing
ai-review .\new_feature.py
ai-analyze .\new_feature.py
Auto-Fix-Code .\new_feature.py
```

---

## 🔥 Advanced Features

### Custom Context Size

Modify in `mamba_ai_v2.ps1`:
```powershell
$MaxFiles = 100      # Default: 50
$MaxFileSizeKB = 200 # Default: 100
```

### Multi-Language Support

Automatically detects:
- **Python**: `*.py`, `requirements.txt`, `setup.py`
- **JavaScript**: `package.json`, `*.js`, `*.jsx`
- **TypeScript**: `tsconfig.json`, `*.ts`, `*.tsx`
- **C#**: `*.csproj`, `*.sln`, `*.cs`
- **Java**: `pom.xml`, `build.gradle`, `*.java`
- **Go**: `go.mod`, `go.sum`, `*.go`
- **Rust**: `Cargo.toml`, `*.rs`
- **C++**: `CMakeLists.txt`, `*.cpp`, `*.h`

### API Model Selection

Default: GPT-4

**Change Model**:
Edit `mamba_ai_v2.ps1`:
```powershell
$script:DEFAULT_MODEL = "gpt-4"          # Most capable
$script:DEFAULT_MODEL = "gpt-3.5-turbo"  # Faster, cheaper
$script:DEFAULT_MODEL = "claude-3-opus-20240229"  # If using Claude
```

---

## 🏆 Why Mamba AI v2 is "Beyond ChatGPT & Claude"

| Feature | ChatGPT/Claude Web | Mamba AI v2 |
|---------|-------------------|-------------|
| **Codebase Access** | ❌ Must paste code | ✅ Reads entire project automatically |
| **GitHub Search** | ❌ No access | ✅ Search repos, read files |
| **Stack Overflow** | ❌ No access | ✅ Search + extract answers |
| **YouTube** | ❌ No access | ✅ Find tutorials |
| **File System** | ❌ No access | ✅ Read/write/analyze files |
| **Code Analysis** | ❌ Manual | ✅ Automated syntax/quality checks |
| **Persistent Memory** | ❌ Session only | ✅ Survives restarts |
| **Security Sandbox** | ❌ N/A | ✅ Protects system files |
| **Multi-Source Research** | ❌ Single source | ✅ Cross-references 5+ sources |
| **Language Detection** | ❌ Manual | ✅ Automatic |

---

## 💡 Pro Tips

1. **Navigate First, Then Ask**:
   ```powershell
   cd C:\MyProject
   ai "explain this"  # Analyzes MyProject
   ```

2. **Use Specific File Paths**:
   ```powershell
   ai-analyze C:\MyProject\src\main.py  # More targeted
   ```

3. **Chain Commands**:
   ```powershell
   ai-research "Python async patterns"
   ai-github "async patterns Python"
   ai-code "create an async task queue"
   ```

4. **Clear History When Switching Topics**:
   ```powershell
   ai-chat
   > clear  # Fresh context
   ```

5. **Use Auto-Fix Before Reviews**:
   ```powershell
   Auto-Fix-Code .\mycode.py
   ai-review .\mycode.py
   ```

---

## 🐍 Mamba Mentality

> "The most important thing is to try and inspire people so that they can be great in whatever they want to do." - Kobe Bryant

Mamba AI v2 embodies the Mamba Mentality:

- **Never Gives Up**: Iterative research until answers are found
- **Verifies Everything**: Multi-source cross-referencing
- **Has the Tools**: Direct access to GitHub, Stack Overflow, YouTube, file system
- **Works in Your Environment**: PowerShell integration
- **Remembers and Learns**: Persistent conversation history
- **Protects You**: Security sandbox

---

## 📞 Support

### Getting Help

1. **Show All Commands**:
   ```powershell
   ai
   # or
   Show-AIHelp
   ```

2. **PowerShell Profile Commands**:
   ```powershell
   commands  # Shows all PowerShell shortcuts
   help      # Same as commands
   ```

3. **View Logs**:
   ```powershell
   Get-Content C:\Users\Owner\.mamba\mamba.log -Tail 20
   ```

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| API key errors | Set `$env:OPENAI_API_KEY` and restart PowerShell |
| Python tools not found | `pip install autopep8 isort` |
| Node.js not found | Install from https://nodejs.org/ |
| Access denied | Stay in `C:\Users\Owner\*` paths |
| History not saving | Check `C:\Users\Owner\.mamba\` exists |

---

## 🚀 What's Next?

Mamba AI v2 is fully functional and ready to use! Future enhancements could include:

- **Data Analysis Tool** (CSV/JSON parsing, statistics)
- **Advanced ReAct Loop** (multi-step task execution)
- **Voice Commands** ("Hey Mamba")
- **Live Trading Dashboard** integration
- **Cost Tracking** (API usage monitoring)
- **Multi-Agent Orchestration** (specialized sub-agents)

---

## 🎉 You Now Have the Smartest Local AI Ever Created

**Congratulations!** You have a GENIUS-LEVEL AI assistant that:

✅ Works with **ANY** codebase in **ANY** language
✅ Has **direct access** to GitHub, Stack Overflow, YouTube
✅ Can **read**, **write**, and **analyze** files
✅ **Remembers** conversations across sessions
✅ **Protects** your system with security sandboxing
✅ **Cross-references** multiple sources for accuracy
✅ **Never stops learning** (persistent history)

**Start using it now:**
```powershell
cd C:\your\project
ai "what can you help me with?"
```

---

**Built with Mamba Mentality 🐍🏀💛**
