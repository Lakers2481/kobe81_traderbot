# /telegram

Manage Telegram bot notifications and alerts.

## Usage
```
/telegram [status|test|config|history]
```

## What it does
1. Configure Telegram bot token and chat ID
2. Send test messages to verify connection
3. Set alert triggers (trades, drawdowns, daily summaries)
4. View message history and delivery status

## Commands
```bash
# Test Telegram connection
python -c "
from monitor.telegram_bot import TelegramNotifier
import os
from dotenv import load_dotenv
load_dotenv('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')
bot = TelegramNotifier()
bot.send_test()
"

# Send custom alert
python -c "
from monitor.telegram_bot import TelegramNotifier
bot = TelegramNotifier()
bot.send('Kobe Alert: Manual test message')
"

# Check config
echo "TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN:0:10}..."
echo "TELEGRAM_CHAT_ID: $TELEGRAM_CHAT_ID"
```

## Alert Triggers
- Trade executed (entry/exit)
- Drawdown exceeds threshold (default: 5%)
- Daily P&L summary (EOD)
- Kill switch activated
- Preflight failure
- Position limit reached

## Environment Variables
```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Setup
1. Create bot via @BotFather on Telegram
2. Get chat ID via @userinfobot
3. Add to .env file
4. Run `/telegram test` to verify


