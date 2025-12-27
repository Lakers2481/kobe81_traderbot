# /performance

Real-time system performance monitoring.

## Usage
```
/performance [--live] [--profile] [--benchmark]
```

## What it does
1. Monitor CPU, memory, disk usage
2. Track API response times
3. Measure execution latency
4. Profile slow functions

## Commands
```bash
# Current performance snapshot
python scripts/performance.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Live monitoring (updates every 5s)
python scripts/performance.py --live

# Profile a script
python scripts/performance.py --profile scripts/run_paper_trade.py

# Benchmark API latency
python scripts/performance.py --benchmark-api

# Memory profiling
python scripts/performance.py --memory-profile
```

## Metrics

### System Resources
| Metric | Current | Limit | Status |
|--------|---------|-------|--------|
| CPU | 12% | 80% | OK |
| Memory | 1.2GB | 4GB | OK |
| Disk | 45GB free | 10GB | OK |

### API Latency
| Endpoint | Avg | P95 | P99 |
|----------|-----|-----|-----|
| Polygon bars | 120ms | 250ms | 450ms |
| Alpaca quote | 45ms | 80ms | 120ms |
| Alpaca order | 85ms | 150ms | 280ms |

### Execution Timing
| Operation | Avg | Max |
|-----------|-----|-----|
| Signal scan | 2.1s | 4.5s |
| Order submit | 0.3s | 0.8s |
| Data fetch (1 sym) | 0.5s | 1.2s |
| Full universe scan | 45s | 90s |

## Output
```
PERFORMANCE DASHBOARD
=====================
Uptime: 4h 32m
Last scan: 2024-12-25 14:30:00

SYSTEM:
  CPU: 12% (4 cores)
  Memory: 1.2GB / 16GB (7.5%)
  Disk: 45GB free

API HEALTH:
  Polygon: 120ms avg [OK]
  Alpaca: 45ms avg [OK]
  Rate limits: 80% remaining

TIMING (last hour):
  Scans: 12 completed
  Avg scan time: 2.1s
  Slowest: 4.5s (14:15:00)

BOTTLENECKS:
  [WARN] Polygon fetch 63% of scan time
  [INFO] Consider batch requests
```

## Alerts
| Condition | Alert |
|-----------|-------|
| CPU > 80% | Warning |
| Memory > 80% | Warning |
| Disk < 10GB | Critical |
| API latency > 1s | Warning |
| Scan time > 60s | Warning |

## Profiling Output
```
PROFILE: run_paper_trade.py
===========================
Total: 12.4s

TOP 5 SLOW FUNCTIONS:
  1. fetch_bars_polygon: 8.2s (66%)
  2. compute_indicators: 2.1s (17%)
  3. generate_signals: 1.2s (10%)
  4. place_order: 0.5s (4%)
  5. log_event: 0.2s (2%)

RECOMMENDATION:
  Cache more aggressively for fetch_bars_polygon
```

## Integration
- Feeds into /status
- Alerts via /telegram
- Logs to logs/performance.jsonl
- Grafana-compatible metrics export


