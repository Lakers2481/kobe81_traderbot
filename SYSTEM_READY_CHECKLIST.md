# 🎯 KOBE TRADING SYSTEM - READY CHECKLIST

## ✅ Complete System Verification

### Step 1: Reload PowerShell

```powershell
reload
```

You should see:
```
🐍 Mamba AI v2 loaded!
🤖 AI: Claude (Anthropic)
🐍 Mamba AI v3 loaded!
💬 Natural chat loaded!
🤝 Kobe Trading Integration loaded!
💬 Natural chat mode loaded!
🎮 Control Panel loaded!
```

---

### Step 2: Run Full System Verification

```powershell
cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot
.\verify_complete_system.ps1
```

**Expected Result:** ✅ SYSTEM READY FOR TRADING!

**If you see failures:**
1. Read the error messages
2. Follow the "Fix:" instructions
3. Run verification again

---

### Step 3: Open Control Panel

```powershell
kobe
```

This opens the master control panel with all trading operations.

---

## 🎮 Master Control Panel Overview

The `kobe` command gives you access to everything:

### Verification & Health (Options 1-4)
- **[1] Full System Verification** ← RUN THIS FIRST!
- [2] Quick Status Check
- [3] Verify Data Quality
- [4] Run Preflight Check

### Backtesting & Analysis (Options 5-8)
- [5] Backtest 5 Years
- [6] Backtest 10 Years
- [7] Walk-Forward Analysis
- [8] Custom Backtest (choose dates)

### Scanning & Signals (Options 9-12)
- [9] Daily Scan (200 stocks)
- [10] Full Scan (800 stocks + Top5)
- [11] Markov-Enhanced Scan
- [12] View Recent Signals

### AI/ML Components (Options 13-16)
- [13] HMM Regime Detection
- [14] Markov Chain Analysis
- [15] LSTM Confidence Model
- [16] Cognitive Brain Status

### Mamba AI Chat (Options 17-20)
- [17] Start Interactive Chat
- [18] Ask Quick Question
- [19] Run Autonomous Task
- [20] Scan Codebase for Issues

### Risk & Safety (Options 21-24)
- [21] Check Risk Gates
- [22] View Kill Switch Status
- [23] Full System Audit
- [24] Verify Hash Chain

---

## 💬 Three Ways to Use Mamba AI

### 1. Control Panel (Interactive Menu)
```powershell
kobe  # Opens control panel
# Select option 17 for chat
```

### 2. Direct Chat Command
```powershell
chat  # Starts interactive chat session
```

Then type naturally:
```
what is my trading bot?
run a 5 year backtest
verify my data
show system status
exit
```

### 3. One-Line Commands
```powershell
talk what files are in this folder?
talk explain my dual strategy
talk run a scan for today
```

---

## 🚀 Daily Workflow

### Morning Routine (Before Market Open)

```powershell
kobe  # Open control panel
```

1. **Option 1**: Full System Verification
2. **Option 3**: Verify Data Quality
3. **Option 10**: Full Scan (800 stocks + Top5)
4. **Option 2**: Quick Status Check

### During Market Hours

```powershell
chat
```

```
show me today's signals
what are my open positions?
analyze AAPL historical patterns
is there a good setup right now?
```

### After Market Close

```powershell
kobe
```

1. **Option 2**: Quick Status Check
2. **Option 12**: View Recent Signals
3. **Option 23**: Full System Audit

---

## 📊 Running Your First Backtest

### Quick Test (5 Years)

```powershell
kobe  # Open control panel
```

Select **Option 5**: Backtest 5 Years

**OR**

```powershell
Run-Backtest -Years 5
```

### Full Test (10 Years)

```powershell
kobe  # Open control panel
```

Select **Option 6**: Backtest 10 Years

**OR**

```powershell
Run-Backtest -Years 10
```

### Custom Date Range

```powershell
kobe  # Open control panel
```

Select **Option 8**: Custom Backtest

Enter start date: `2020-01-01`
Enter end date: `2024-12-31`

**OR**

```powershell
Run-Backtest -Start "2020-01-01" -End "2024-12-31"
```

---

## 🔎 Running Scans

### Daily Scan (Quick)

```powershell
Run-Scan
```

### Full Universe Scan with Top 5

```powershell
Run-Scan -Cap 800 -Top5 -Deterministic
```

### With Markov Chain Integration

```powershell
Run-Scan -Cap 800 -Top5 -WithMarkov -Deterministic
```

**Output Files:**
- `logs/daily_top5.csv` - Top 5 stocks to STUDY
- `logs/tradeable.csv` - Top 2 stocks to TRADE
- `logs/signals.jsonl` - All raw signals

---

## 🤖 Using AI/ML Components

### Check Market Regime

```powershell
Run-AIComponent -Component hmm
```

### Get Direction Prediction

```powershell
Run-AIComponent -Component markov
```

### Check Signal Confidence

```powershell
Run-AIComponent -Component lstm
```

### Check Cognitive Brain

```powershell
Run-AIComponent -Component brain
```

---

## 📈 Historical Analysis

### Analyze Single Stock

```powershell
Analyze-HistoricalData -Symbol AAPL -Years 5
```

### Analyze Multiple Stocks

```powershell
chat
```

```
analyze historical patterns for AAPL, TSLA, and NVDA over 5 years
compare their reversal rates
which one has the best consecutive day patterns?
```

---

## 🔒 Safety Checks

### Before Trading

```powershell
kobe  # Open control panel
```

1. **Option 1**: Full System Verification
2. **Option 21**: Check Risk Gates
3. **Option 22**: View Kill Switch Status
4. **Option 4**: Run Preflight Check

### Before Going Live

```powershell
kobe  # Open control panel
```

1. **Option 23**: Full System Audit
2. **Option 24**: Verify Hash Chain
3. **Option 3**: Verify Data Quality

**ALL THREE MUST PASS** before live trading!

---

## 🐛 Troubleshooting

### "Control panel not working"

```powershell
reload
kobe
```

### "Commands not found"

```powershell
Get-Command Run-Backtest
Get-Command Run-Scan
Get-Command chat
```

If not found:
```powershell
reload
```

### "API key errors"

```powershell
$env:ANTHROPIC_API_KEY  # Should show key
$env:POLYGON_API_KEY    # Should show key
$env:ALPACA_API_KEY_ID  # Should show key
```

If empty:
```powershell
reload
```

### "Python errors"

```powershell
python --version  # Should be 3.11+
pip list | grep pandas
pip list | grep numpy
```

If missing:
```powershell
cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot
pip install -r requirements.txt
```

---

## ✅ Pre-Flight Checklist (MUST COMPLETE)

Before trading, verify:

- [ ] Full system verification passes (`.\verify_complete_system.ps1`)
- [ ] All API keys loaded (Polygon, Alpaca, Claude)
- [ ] Universe file exists (800 stocks)
- [ ] Python 3.11+ installed
- [ ] All required packages installed
- [ ] Risk gates functional
- [ ] Kill switch is OFF
- [ ] Data is fresh and validated
- [ ] Backtest passes (5-10 years)
- [ ] Control panel accessible (`kobe` works)
- [ ] Chat works (`chat` works)

---

## 🎯 Quick Command Reference

| What You Want | Command |
|---------------|---------|
| **Open Control Panel** | `kobe` |
| **Interactive Chat** | `chat` |
| **Quick Question** | `talk your question here` |
| **Verify System** | `.\verify_complete_system.ps1` |
| **Backtest 5 Years** | `Run-Backtest -Years 5` |
| **Scan Market** | `Run-Scan` |
| **Full Scan** | `Run-Scan -Cap 800 -Top5 -WithMarkov` |
| **System Status** | `Show-KobeStatus` |
| **Data Quality** | `Verify-Data` |
| **Full Audit** | `Run-FullAudit` |
| **AI Component** | `Run-AIComponent -Component hmm` |
| **Historical** | `Analyze-HistoricalData -Symbol AAPL` |

---

## 🎉 You're Ready When...

✅ Verification script passes with 0 failures
✅ Control panel opens without errors
✅ Chat responds to questions
✅ Backtest completes successfully
✅ Scan produces signals
✅ All risk gates functional
✅ Kill switch is OFF
✅ Data is validated and fresh

**Type `kobe` and select Option 1 to verify everything!**

---

**Built with Mamba Mentality 🐍🏀💛**
