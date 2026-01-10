# 🚀 AI SETUP - QUICK START CARD

## Your Current Situation

```
❌ Claude API: Has key but NO CREDITS
⚠️  OpenAI API: Placeholder added, needs real key
```

---

## ⚡ FASTEST PATH TO WORKING AI (Pick One)

### OPTION A: Fix Claude (5 minutes)
```
1. Visit: https://console.anthropic.com/settings/billing
2. Add $20 credits
3. Run: reload
4. Test: talk hello

✅ DONE! Start chatting.
```

**Best for**: Trading analysis, complex questions
**Cost**: $20 lasts 2-3 months (light use)

---

### OPTION B: Add OpenAI (5 minutes)
```
1. Visit: https://platform.openai.com/api-keys
2. Create account + add $5 credits
3. Copy API key (sk-proj-...)
4. Open: code C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\.env
5. Replace: OPENAI_API_KEY=your_openai_key_here_replace_this
   With: OPENAI_API_KEY=sk-proj-YOUR_ACTUAL_KEY
6. Run: reload
7. Test: talk hello

✅ DONE! Start chatting.
```

**Best for**: Quick answers, faster responses
**Cost**: $5 lasts 1-2 months (light use)

---

## 🏆 RECOMMENDED: BOTH (10 minutes)

Do BOTH Option A and Option B for:
- ✅ Automatic failover if one runs out
- ✅ Best of both worlds
- ✅ Never blocked from AI access

**Total cost**: $25 for months of usage

---

## 📋 CHECKLIST

After setup, verify:

```powershell
# Check your setup status
cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot
.\check_ai_setup.ps1
```

Expected output:
```
🎉 PERFECT SETUP! Both AI providers working!
   Primary: Claude (Anthropic)
   Fallback: OpenAI (GPT-4)

✅ You're ready to chat!
   Try: talk hello
```

---

## 🧪 TEST YOUR AI

### Test 1: Simple Question
```powershell
talk hello, what AI am I using?
```

### Test 2: Code Analysis
```powershell
cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot
talk what files are in this folder?
```

### Test 3: Trading Knowledge
```powershell
talk explain the IBS RSI strategy
```

### Test 4: Interactive Chat
```powershell
chat
```
Then type naturally:
```
what is mean reversion?
how does the dual strategy work?
exit
```

---

## 🔗 DIRECT LINKS

| Service | Purpose | URL |
|---------|---------|-----|
| **Claude Billing** | Add credits | https://console.anthropic.com/settings/billing |
| **Claude Console** | Manage account | https://console.anthropic.com/ |
| **OpenAI API Keys** | Create key | https://platform.openai.com/api-keys |
| **OpenAI Billing** | Add credits | https://platform.openai.com/settings/organization/billing |

---

## 💡 WHICH SHOULD I CHOOSE?

| Scenario | Recommendation |
|----------|----------------|
| "I want the best trading analysis" | ⭐ **Claude** (more analytical) |
| "I want faster responses" | ⭐ **OpenAI** (faster, cheaper) |
| "I want both and auto-failover" | ⭐⭐⭐ **BOTH** (recommended) |
| "I'm on a tight budget" | OpenAI ($5) or Claude ($20) |
| "Money is not an issue" | **BOTH** ($25 total) |

---

## 🆘 HELP!

### Issue: "Still getting credit error"
```powershell
# Wait 5 minutes after payment
# Then reload
reload
talk hello
```

### Issue: "Wrong AI provider"
```powershell
# Check which is loaded
$env:ANTHROPIC_API_KEY    # Shows Claude key
$env:OPENAI_API_KEY       # Shows OpenAI key

# To force OpenAI:
# 1. Open .env
code C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\.env

# 2. Comment out Claude:
# ANTHROPIC_API_KEY=sk-ant-...

# 3. Reload
reload
```

### Issue: "Can't find .env file"
```powershell
# Open in VS Code
code C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\.env

# Or in Notepad
notepad C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\.env
```

---

## 📚 FULL GUIDE

For complete documentation, see:
**`AI_SETUP_COMPLETE_GUIDE.md`**

---

## ✅ YOU'RE READY WHEN...

You see this after reload:
```
🐍 Mamba AI v2 loaded! Type ai for help
📂 Current: C:\Users\Owner
🤖 AI: Claude (Anthropic)        ← or ← OpenAI

💬 Simple chat loaded! Type talk to start chatting
```

And this works:
```powershell
talk hello, test message
# ✅ Gets AI response (not error)
```

---

**Built with Mamba Mentality 🐍🏀💛**

**NEXT STEP**: Pick Option A or B above and get started! 🚀
