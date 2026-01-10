# 🤖 COMPLETE AI SETUP GUIDE - ALL OPTIONS

## Current Status

❌ **Claude (Anthropic)**: API key exists but has no credits
❓ **OpenAI (GPT-4)**: No API key configured yet

---

## OPTION 1: Add Credits to Claude (Recommended for Trading)

Claude Opus is more analytical and better for trading decisions.

### Step 1: Visit Anthropic Console
1. Open: **https://console.anthropic.com/**
2. Log in with your account

### Step 2: Navigate to Billing
1. Click your profile (top right)
2. Select **"Settings"**
3. Click **"Plans & Billing"**

### Step 3: Add Credits
1. Click **"Purchase Credits"** or **"Add Payment Method"**
2. Options:
   - **Pay-as-you-go**: Add $20-$100 credits
   - **Monthly plan**: $20-$100/month

### Pricing Reference (Claude Opus)
- Input: $15 per million tokens (~750,000 words)
- Output: $75 per million tokens (~750,000 words)
- **For trading use**: $20 should last weeks of daily chat

### Step 4: Verify
After adding credits, test:
```powershell
reload
talk hello, test my claude connection
```

If you see a response, it's working! ✅

---

## OPTION 2: Add OpenAI as Fallback (Best Setup)

This gives you automatic failover if Claude runs out of credits.

### Step 1: Get OpenAI API Key

1. Visit: **https://platform.openai.com/signup**
2. Create account (or log in)
3. Add payment method:
   - Go to **Settings** → **Billing** → **Add payment method**
   - Add at least $5 credit

4. Create API key:
   - Go to **API Keys** (https://platform.openai.com/api-keys)
   - Click **"Create new secret key"**
   - Name it: "Mamba AI"
   - Copy the key (starts with `sk-proj-...` or `sk-...`)
   - **IMPORTANT**: Save it immediately - you can't see it again!

### Pricing Reference (GPT-4)
- GPT-4: $2.50 per million input tokens
- GPT-4: $10.00 per million output tokens
- **For trading use**: $5 should last a month of daily chat

### Step 2: Add to .env File

I'll add this for you automatically. Your `.env` will have:

```
# CLAUDE API (Cognitive Layer / AI Reasoning)
ANTHROPIC_API_KEY=your_anthropic_key_here_replace_this

# OPENAI API (Fallback AI Provider)
OPENAI_API_KEY=your_openai_key_here_replace_this
```

### Step 3: Update the Key

After you get your OpenAI key:
```powershell
# Open the .env file
code C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\.env

# Replace the line:
OPENAI_API_KEY=your_openai_key_here_replace_this

# With your actual key:
OPENAI_API_KEY=sk-proj-YOUR_ACTUAL_KEY_HERE
```

### Step 4: Reload
```powershell
reload
```

### Step 5: Test Fallback
```powershell
# If Claude has no credits, system will automatically use OpenAI
talk hello, which AI am I talking to?
```

---

## OPTION 3: Switch to OpenAI as Primary

If you want to use OpenAI instead of Claude:

### Method A: Disable Claude Temporarily
```powershell
# Open .env
code C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot\.env

# Comment out Claude key:
# ANTHROPIC_API_KEY=sk-ant-api03-...

# Reload
reload
```

Now the system will use OpenAI by default.

### Method B: Remove Claude Key Entirely
Same as above, but delete the line instead of commenting it.

### Verify Which Provider is Active
After reload, you should see:
```
🤖 AI: GPT-4 (OpenAI)          ← Shows which AI is loaded
```

Or:
```
🤖 AI: Claude (Anthropic)      ← Shows which AI is loaded
```

---

## How the Auto-Detection Works

Your Mamba AI system is smart:

```
┌─────────────────────────────────────────────────────────┐
│  PROVIDER DETECTION LOGIC                               │
├─────────────────────────────────────────────────────────┤
│  1. Check for ANTHROPIC_API_KEY in .env                 │
│     ✅ Found → Set default to Claude Opus               │
│     ❌ Not found → Continue to step 2                   │
│                                                          │
│  2. Check for OPENAI_API_KEY in .env                    │
│     ✅ Found → Set default to GPT-4                     │
│     ❌ Not found → Set default to GPT-4 (will fail)     │
│                                                          │
│  3. If API call fails → Try fallback provider           │
│     (Claude fails → Try OpenAI if available)            │
│     (OpenAI fails → Try Claude if available)            │
└─────────────────────────────────────────────────────────┘
```

**Location**: `mamba_ai_v2.ps1` lines 98-106

---

## Recommended Setup (Best of All Worlds)

1. ✅ **Add $20 to Claude** (primary - best for trading analysis)
2. ✅ **Add $5 to OpenAI** (fallback - never get blocked)
3. ✅ **Keep both keys in .env** (automatic failover)

### Why This is Best:
- Claude is more analytical for trading decisions
- OpenAI is faster for simple questions
- If one service is down, you have backup
- Total cost: $25 for months of AI chat

---

## Testing Your Setup

### Test 1: Check What's Loaded
```powershell
reload
# Look for this line in the output:
🤖 AI: Claude (Anthropic)   ← or ← 🤖 AI: GPT-4 (OpenAI)
```

### Test 2: Simple Chat
```powershell
talk hello, what AI provider am I using?
```

### Test 3: Code Analysis
```powershell
cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot
talk what files are in this folder?
```

### Test 4: Trading Question
```powershell
talk explain mean reversion trading
```

### Test 5: Interactive Chat
```powershell
chat
```

Then type:
```
what is the dual strategy scanner?
how does the turtle soup strategy work?
exit
```

---

## Troubleshooting

### Issue: "API key not loaded"
```powershell
# Check if key is in environment
$env:ANTHROPIC_API_KEY
$env:OPENAI_API_KEY

# If empty, reload
reload
```

### Issue: "Still getting credit error"
- Wait 5 minutes after adding credits
- Check billing page to confirm payment processed
- Try reloading: `reload`

### Issue: "Neither API works"
- Verify both keys are correct in `.env`
- Check for typos (no spaces, no quotes)
- Ensure payment methods are active on both platforms

### Issue: "Wrong AI provider"
```powershell
# Check which keys are set
$env:ANTHROPIC_API_KEY    # Should show sk-ant-...
$env:OPENAI_API_KEY       # Should show sk-proj-... or sk-...

# To force OpenAI, comment out Claude in .env:
# ANTHROPIC_API_KEY=...

# Reload
reload
```

---

## Quick Reference: Commands

| Command | What It Does |
|---------|--------------|
| `talk <question>` | Ask AI one question |
| `chat` | Start interactive chat session |
| `ai <question>` | Ask AI anything (verbose mode) |
| `ai-chat` | Interactive AI with history |
| `reload` | Reload PowerShell profile + API keys |
| `kobe` | Open control panel |
| `commands` | Show all available commands |

---

## Cost Estimates

### Claude (Anthropic)
| Usage | Cost | Lasts |
|-------|------|-------|
| Light (10 questions/day) | $20 | 2-3 months |
| Medium (50 questions/day) | $20 | 3-4 weeks |
| Heavy (200 questions/day) | $20 | 1 week |

### OpenAI (GPT-4)
| Usage | Cost | Lasts |
|-------|------|-------|
| Light (10 questions/day) | $5 | 1-2 months |
| Medium (50 questions/day) | $5 | 2-3 weeks |
| Heavy (200 questions/day) | $5 | 1 week |

**Note**: Trading bot operations (backtests, scans) use Python code, not AI APIs - those are free!

---

## Security Notes

- ✅ `.env` file is in `.gitignore` - won't be committed to git
- ✅ API keys are loaded into PowerShell session only (Process scope)
- ✅ Keys are not stored in Windows registry
- ⚠️ Don't share your `.env` file with anyone
- ⚠️ Don't commit `.env` to GitHub/GitLab

---

## Next Steps

1. **Now**: Add OpenAI key placeholder to `.env` (I'll do this)
2. **You**: Get OpenAI API key from platform.openai.com
3. **You**: Add credits to Claude at console.anthropic.com
4. **You**: Update `.env` with your OpenAI key
5. **You**: Run `reload` and test with `talk hello`

---

**Built with Mamba Mentality 🐍🏀💛**
