# /journal

Trading journal for notes, lessons, and trade reviews.

## Usage
```
/journal [add|view|search|review]
```

## What it does
1. Log trading observations and lessons
2. Review specific trades with notes
3. Track emotional/behavioral patterns
4. Search past entries

## Commands
```bash
# Add journal entry
python scripts/journal.py add --note "RSI-2 working well in choppy market" --tags market,rsi2

# View recent entries
python scripts/journal.py view --days 7

# Search entries
python scripts/journal.py search --query "drawdown"

# Review specific trade
python scripts/journal.py review --trade-id DEC_20241225_AAPL_ABC123

# Daily reflection
python scripts/journal.py add --type daily --note "Good discipline today, followed all rules"

# Tag a lesson learned
python scripts/journal.py add --type lesson --note "Don't chase after missing entry" --tags discipline
```

## Entry Types
| Type | Purpose |
|------|---------|
| observation | Market/strategy observations |
| lesson | Lessons learned |
| daily | End-of-day reflection |
| review | Trade-specific review |
| idea | Future research ideas |

## Journal Location
- `logs/journal.jsonl` - All entries (JSON lines)
- `logs/journal_weekly.md` - Weekly summary (generated)

## Best Practices
1. Write daily EOD reflection
2. Review all losing trades
3. Note what worked and why
4. Track recurring mistakes
5. Document strategy tweaks

## Weekly Review
```bash
# Generate weekly summary
python scripts/journal.py weekly --output logs/journal_weekly.md
```
