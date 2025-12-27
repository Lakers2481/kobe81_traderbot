# /dashboard

Launch and manage the Kobe trading dashboard.

## Usage
```
/dashboard [start|stop|status|refresh]
```

## What it does
1. Start/stop the web dashboard
2. Display real-time trading status
3. Show positions, P&L, signals
4. Monitor system health

## Commands
```bash
# Start dashboard server
python scripts/dashboard.py --port 8080 --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Start in background
python scripts/dashboard.py --port 8080 --daemon

# Stop dashboard
python scripts/dashboard.py --stop

# Check if running
python scripts/dashboard.py --status

# Refresh data only (no restart)
python scripts/dashboard.py --refresh
```

## Dashboard Panels

### 1. System Status
- Mode: PAPER / LIVE / HALTED
- Kill switch state
- Last scan time
- Data freshness
- Broker connection

### 2. Positions
- Open positions table
- Entry price, current price
- Unrealized P&L ($ and %)
- Days held
- Stop loss distance

### 3. P&L Summary
- Today's P&L
- Week P&L
- Month P&L
- Total P&L
- Win rate / Profit factor

### 4. Signals
- Latest scan results
- Pending signals
- Signal history (24h)
- Strategy breakdown

### 5. Risk Metrics
- Current exposure
- Sector allocation
- Correlation matrix
- Drawdown status

### 6. System Health
- API status (Polygon, Alpaca)
- Data cache age
- Last integrity check
- Error count (24h)

## Access
```
http://localhost:8080
```

## Troubleshooting

### Dashboard won't start
```bash
# Check port in use
netstat -ano | findstr :8080

# Kill process on port
taskkill /F /PID <pid>

# Try different port
python scripts/dashboard.py --port 8888
```

### Data not updating
```bash
# Force refresh
python scripts/dashboard.py --refresh

# Check data freshness
/data

# Refetch if stale
/prefetch
```

### Slow performance
```bash
# Reduce refresh rate
python scripts/dashboard.py --refresh-interval 30

# Check logs for errors
/logs --tail 50
```

## Configuration
```json
{
  "dashboard": {
    "port": 8080,
    "refresh_interval": 10,
    "auth_enabled": false,
    "dark_mode": true
  }
}
```

## Integration
- Auto-starts with /start
- Stops with /stop
- Health endpoint: /health includes dashboard status
- Logs to logs/dashboard.log


