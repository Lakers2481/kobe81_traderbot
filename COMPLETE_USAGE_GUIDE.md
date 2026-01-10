# 🎉 COMPLETE MAMBA AI USAGE GUIDE

## ✅ Three Ways to Chat with Mamba AI

### 1. INTERACTIVE CHAT MODE (Best for conversations!)

Type `chat` and enter interactive mode:

```powershell
chat
```

Then just type naturally:
```
You: what are you doing?
Mamba AI: I'm here to help you with your Kobe trading bot...

You: explain my trading strategies
Mamba AI: Your bot uses two main strategies...

You: run a backtest for 5 years
Mamba AI: Let me run that backtest...

You: exit
```

**This is the EASIEST way!** No syntax to remember.

---

### 2. ONE-LINE CHAT (Quick questions)

Use `talk` followed by your question **ON THE SAME LINE**:

```powershell
talk what are you doing?
talk explain this code
talk find all Python files
talk what is my trading bot?
```

**Important:** Type `talk` AND your question together!

```powershell
# ✅ CORRECT
talk what is 2 plus 2?

# ❌ WRONG - Don't press Enter after "talk"!
talk
what is 2 plus 2?
```

---

### 3. TRADITIONAL AI (With quotes)

```powershell
ai "what is 2 plus 2?"
ai "explain my trading system"
ai "find all bugs"
```

---

## 🤖 Full Trading Bot Control

### Backtesting

```powershell
# Run backtest for last 5 years
Run-Backtest -Years 5

# Run backtest for last 10 years
Run-Backtest -Years 10

# Run backtest with specific dates
Run-Backtest -Start "2020-01-01" -End "2024-12-31"

# Run backtest on 200 stocks
Run-Backtest -Years 5 -Cap 200

# Walk-forward analysis (train/test splits)
Run-WalkForward -Years 10
```

### Scanning

```powershell
# Daily scan (200 stocks)
Run-Scan

# Full universe scan (800 stocks)
Run-Scan -Cap 800

# Scan with Top 5 output
Run-Scan -Cap 800 -Top5

# Scan with Markov chain integration
Run-Scan -Cap 800 -Top5 -WithMarkov

# Deterministic scan (reproducible)
Run-Scan -Cap 800 -Deterministic -Top5
```

### Data Verification

```powershell
# Verify all data quality
Verify-Data

# Run full system audit
Run-FullAudit
```

### AI/ML Components

```powershell
# Run HMM regime detection
Run-AIComponent -Component hmm

# Run Markov chain analysis
Run-AIComponent -Component markov

# Run LSTM confidence model
Run-AIComponent -Component lstm

# Run ensemble predictor
Run-AIComponent -Component ensemble

# Check cognitive brain status
Run-AIComponent -Component brain

# Check research OS status
Run-AIComponent -Component research
```

### Historical Analysis

```powershell
# Analyze historical patterns for AAPL
Analyze-HistoricalData -Symbol AAPL -Years 5

# Analyze TSLA over 10 years
Analyze-HistoricalData -Symbol TSLA -Years 10
```

### System Status

```powershell
# Show comprehensive system status
Show-KobeStatus

# Show all trading commands
Show-KobeCommands
```

---

## 💬 Natural Language Trading Commands

You can also use natural language with the AI to control your trading bot!

### In Interactive Chat Mode:

```powershell
chat
```

Then type:

```
You: run a backtest for the last 5 years
Mamba AI: [Executes Run-Backtest -Years 5]

You: scan all 800 stocks with markov integration
Mamba AI: [Executes Run-Scan -Cap 800 -WithMarkov]

You: verify my data quality
Mamba AI: [Executes Verify-Data]

You: show me system status
Mamba AI: [Executes Show-KobeStatus]

You: analyze historical data for AAPL
Mamba AI: [Executes Analyze-HistoricalData -Symbol AAPL]
```

### With One-Line Commands:

```powershell
talk run a 10 year walk-forward backtest
talk scan the market with top 5 output
talk run the HMM regime detector
talk show me all open positions
talk audit the entire system
```

---

## 🎯 Real-World Examples

### Example 1: Daily Morning Routine

```powershell
# Start chat
chat

# Then type:
good morning, verify my data is up to date
show me system status
scan the market for today's setups
what are my open positions?
```

### Example 2: Research New Strategy

```powershell
# Interactive mode
chat

# Conversation:
I want to test a new momentum strategy
run a 5 year backtest on 800 stocks
analyze historical patterns for NVDA
show me the results
what's the win rate?
```

### Example 3: System Health Check

```powershell
# One-line commands
talk verify all data quality
talk run full system audit
talk check cognitive brain status
talk show me any recent errors
```

### Example 4: Complex Analysis

```powershell
# Interactive session
chat

# Conversation:
run walk-forward analysis for 10 years
analyze AAPL historical patterns over 5 years
run the ensemble ML predictor
what does the HMM regime detector say?
combine all these results and give me a trade recommendation
```

---

## 🔥 Advanced: AI Controls Everything

The AI can understand and execute **ANY** trading bot command:

```powershell
chat
```

```
You: I want to backtest my dual strategy on the full 800 stock universe for the last 10 years, then run a walk-forward analysis with 252 day training windows and 63 day test windows, then show me the Sharpe ratio and win rate

Mamba AI: I'll break this down into steps:
1. Running backtest on 800 stocks for 10 years...
   [Executes Run-Backtest -Years 10 -Cap 800]

2. Running walk-forward analysis...
   [Executes Run-WalkForward -Years 10 -TrainDays 252 -TestDays 63]

3. Analyzing results...
   [Parses output and extracts Sharpe and win rate]

Results:
- Sharpe Ratio: 1.85
- Win Rate: 64.2%
- Profit Factor: 1.60
```

---

## 🎓 Pro Tips

### 1. Use Interactive Mode for Complex Tasks

```powershell
chat
# Then have a conversation - the AI remembers context!
```

### 2. Use One-Line for Quick Commands

```powershell
talk show me system status
talk scan the market
```

### 3. Combine AI Questions with Trading Commands

```powershell
chat
```

```
You: explain my dual strategy
Mamba AI: [Explains the strategy]

You: now run a backtest of that strategy for 5 years
Mamba AI: [Executes backtest]

You: what's the win rate?
Mamba AI: [Analyzes results]

You: is that good?
Mamba AI: [Provides analysis]
```

---

## 🐛 Troubleshooting

### "The term 'what' is not recognized"

**Problem:** You typed commands on separate lines.

```powershell
# ❌ WRONG
talk
what are you doing?

# ✅ CORRECT
talk what are you doing?
```

**Solution:** Type them together, OR use interactive mode:

```powershell
chat
# Then type naturally
```

### "Nothing happens when I type talk"

If you just type `talk` with nothing after it, you'll see help. That's normal!

To actually chat, type:
```powershell
talk hello
```

Or use interactive mode:
```powershell
chat
```

### "AI isn't responding"

1. Check API key is loaded:
```powershell
$env:ANTHROPIC_API_KEY
```

2. Reload PowerShell:
```powershell
reload
```

3. Try the test:
```powershell
talk what is 2 plus 2?
```

---

## 📋 Command Quick Reference

| What You Want | Command |
|---------------|---------|
| **Interactive chat** | `chat` |
| **Quick question** | `talk your question here` |
| **Traditional AI** | `ai "your question"` |
| **5-year backtest** | `Run-Backtest -Years 5` |
| **Scan market** | `Run-Scan` |
| **Scan with Markov** | `Run-Scan -Cap 800 -WithMarkov` |
| **Verify data** | `Verify-Data` |
| **Full audit** | `Run-FullAudit` |
| **Run AI component** | `Run-AIComponent -Component hmm` |
| **System status** | `Show-KobeStatus` |
| **Historical analysis** | `Analyze-HistoricalData -Symbol AAPL` |
| **Walk-forward** | `Run-WalkForward -Years 10` |

---

## 🎯 Your First Task

**Reload PowerShell and try this:**

```powershell
reload
```

**Then start interactive chat:**

```powershell
chat
```

**Then type:**

```
hello, what can you help me with?
explain my Kobe trading bot
show me system status
run a simple scan
exit
```

---

## 🐍 You're Now a Trading Bot Master!

With Mamba AI + Kobe Trading Bot, you have:
- ✅ Natural language control of entire trading system
- ✅ Interactive chat sessions
- ✅ One-line commands
- ✅ Full backtesting (any timeframe)
- ✅ Market scanning (800 stocks)
- ✅ Data verification
- ✅ AI/ML component control
- ✅ Historical analysis
- ✅ System audits
- ✅ Real-time status monitoring

**Just type `chat` and start trading!** 💬📈

---

**Built with Mamba Mentality 🐍🏀💛**
