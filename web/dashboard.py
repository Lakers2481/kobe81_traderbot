"""
Kobe Trading Dashboard - Real-Time Monitoring.

Bloomberg Terminal-style dashboard with 5-second WebSocket refresh.
Monitors positions, signals, kill switches, and market context.

URL: http://localhost:8080
"""

import asyncio
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from web.data_provider import get_data_provider

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Kobe Trading Dashboard",
    version="1.0.0",
    description="Real-Time Trading Monitoring Dashboard"
)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATE_PATH = Path(__file__).parent / "templates" / "dashboard.html"


# =============================================================================
# WEBSOCKET CONNECTION MANAGER
# =============================================================================

class ConnectionManager:
    """Manage WebSocket connections for live updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, data: dict):
        """Send data to all connected clients."""
        async with self._lock:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(data)
                except Exception:
                    disconnected.append(connection)

            for conn in disconnected:
                if conn in self.active_connections:
                    self.active_connections.remove(conn)


manager = ConnectionManager()


# =============================================================================
# API ROUTES
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve dashboard HTML."""
    return get_dashboard_html()


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    provider = get_data_provider()
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "alpaca_connected": provider.check_alpaca_connection(),
        "polygon_connected": provider.check_polygon_connection(),
    })


@app.get("/api/data")
async def get_all_data():
    """Get complete dashboard data bundle."""
    provider = get_data_provider()
    data = provider.get_full_dashboard_data()
    return JSONResponse(data)


@app.get("/api/positions")
async def get_positions():
    """Get current positions."""
    provider = get_data_provider()
    positions = provider.get_positions()
    return JSONResponse(positions)


@app.get("/api/signals")
async def get_signals():
    """Get recent signals."""
    provider = get_data_provider()
    signals = provider.load_signals()
    return JSONResponse([s.to_dict() for s in signals])


@app.get("/api/performance")
async def get_performance():
    """Get performance metrics."""
    provider = get_data_provider()
    metrics = provider.get_performance_metrics()
    return JSONResponse(metrics.to_dict())


@app.get("/api/market")
async def get_market():
    """Get market context."""
    provider = get_data_provider()
    context = provider.get_market_context()
    return JSONResponse(context.to_dict())


@app.get("/api/kill-switches")
async def get_kill_switches():
    """Get kill switch status."""
    provider = get_data_provider()
    metrics = provider.get_performance_metrics()
    status = provider.get_kill_switch_status(metrics)
    return JSONResponse(status.to_dict())


@app.post("/api/kill-switch/activate")
async def activate_kill_switch():
    """Activate kill switch."""
    kill_file = PROJECT_ROOT / "state" / "KILL_SWITCH"
    kill_file.parent.mkdir(parents=True, exist_ok=True)
    kill_file.write_text(f"Activated via dashboard at {datetime.now().isoformat()}")
    return JSONResponse({"status": "activated", "timestamp": datetime.now().isoformat()})


@app.post("/api/kill-switch/deactivate")
async def deactivate_kill_switch():
    """Deactivate kill switch."""
    kill_file = PROJECT_ROOT / "state" / "KILL_SWITCH"
    if kill_file.exists():
        kill_file.unlink()
    return JSONResponse({"status": "deactivated", "timestamp": datetime.now().isoformat()})


@app.get("/api/ml-confidence")
async def get_ml_confidence():
    """Get ML model confidence and status dashboard."""
    try:
        from dashboard.ml_confidence import get_ml_confidence_dashboard
        dashboard = get_ml_confidence_dashboard()
        return JSONResponse(dashboard.to_dict())
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "overall_health": "unknown",
            "components": []
        }, status_code=500)


@app.websocket("/ws/live")
async def websocket_live_updates(websocket: WebSocket):
    """WebSocket endpoint for 5-second live updates."""
    await manager.connect(websocket)
    provider = get_data_provider()

    try:
        # Send initial data
        initial_data = provider.get_full_dashboard_data()
        await websocket.send_json({
            "type": "initial",
            "timestamp": datetime.now().isoformat(),
            "data": initial_data
        })

        # Keep connection alive with 5-second updates
        while True:
            await asyncio.sleep(5)

            # Get fresh data
            update_data = provider.get_full_dashboard_data()

            # Send to this client
            await websocket.send_json({
                "type": "update",
                "timestamp": datetime.now().isoformat(),
                "data": update_data
            })

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await manager.disconnect(websocket)


# =============================================================================
# HTML TEMPLATE
# =============================================================================

def get_dashboard_html() -> str:
    """Load dashboard HTML from template file or return inline version."""
    if TEMPLATE_PATH.exists():
        try:
            return TEMPLATE_PATH.read_text(encoding="utf-8")
        except Exception:
            pass

    # Inline fallback template
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kobe Trading Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0a0a;
            --bg-secondary: #111111;
            --bg-card: #1a1a1a;
            --border: #2a2a2a;
            --text-white: #ffffff;
            --text-gray: #888888;
            --text-muted: #555555;
            --green: #00ff88;
            --red: #ff4444;
            --yellow: #ffaa00;
            --blue: #00aaff;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            color: var(--text-white);
            min-height: 100vh;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .header h1 {
            font-size: 24px;
            font-weight: 700;
            color: var(--green);
        }
        .header .status {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        .status-dot.green { background: var(--green); }
        .status-dot.red { background: var(--red); }
        .status-dot.yellow { background: var(--yellow); }
        .grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 15px;
        }
        .card-title {
            font-size: 12px;
            color: var(--text-gray);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .card-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 28px;
            font-weight: 600;
        }
        .card-value.positive { color: var(--green); }
        .card-value.negative { color: var(--red); }
        .card-value.neutral { color: var(--text-white); }
        .card-subtitle {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }
        .section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 14px;
            color: var(--text-gray);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border);
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
            font-size: 13px;
        }
        th {
            color: var(--text-gray);
            font-weight: 500;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }
        td {
            font-family: 'JetBrains Mono', monospace;
        }
        .kill-switch-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }
        .kill-layer {
            background: var(--bg-secondary);
            border-radius: 6px;
            padding: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .kill-layer.triggered {
            background: rgba(255, 68, 68, 0.15);
            border: 1px solid var(--red);
        }
        .kill-layer-name {
            font-size: 12px;
            color: var(--text-gray);
        }
        .kill-layer-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            font-weight: 600;
        }
        .market-indices {
            display: flex;
            gap: 20px;
        }
        .index-card {
            flex: 1;
            text-align: center;
            padding: 15px;
            background: var(--bg-secondary);
            border-radius: 6px;
        }
        .index-name {
            font-size: 12px;
            color: var(--text-gray);
            margin-bottom: 5px;
        }
        .index-price {
            font-family: 'JetBrains Mono', monospace;
            font-size: 20px;
            font-weight: 600;
        }
        .footer {
            text-align: center;
            color: var(--text-muted);
            font-size: 12px;
            padding: 20px;
        }
        .ws-status {
            font-size: 11px;
            color: var(--text-muted);
        }
        .kill-btn {
            background: var(--red);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
        }
        .kill-btn:hover {
            background: #ff6666;
        }
        .kill-btn.deactivate {
            background: var(--green);
        }
    </style>
</head>
<body>
    <div id="app">
        <div class="header">
            <h1>KOBE TRADING DASHBOARD</h1>
            <div class="status">
                <div class="status-indicator">
                    <span class="status-dot" :class="data.system_healthy ? 'green' : 'red'"></span>
                    <span>{{ data.system_healthy ? 'ALL SYSTEMS GO' : 'ALERT' }}</span>
                </div>
                <div class="ws-status">Last Update: {{ lastUpdate }}</div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="card-title">Account Equity</div>
                <div class="card-value neutral">${{ formatNumber(data.performance?.current_equity || 0) }}</div>
                <div class="card-subtitle">Initial: ${{ formatNumber(data.performance?.initial_capital || 100000) }}</div>
            </div>
            <div class="card">
                <div class="card-title">Cumulative P&L</div>
                <div class="card-value" :class="getPnlClass(data.performance?.cumulative_pnl || 0)">
                    {{ formatPnl(data.performance?.cumulative_pnl || 0) }}
                </div>
                <div class="card-subtitle">{{ data.performance?.cumulative_pnl_pct || 0 }}%</div>
            </div>
            <div class="card">
                <div class="card-title">Win Rate</div>
                <div class="card-value" :class="data.performance?.win_rate >= 50 ? 'positive' : 'negative'">
                    {{ data.performance?.win_rate || 0 }}%
                </div>
                <div class="card-subtitle">{{ data.performance?.total_trades || 0 }} trades</div>
            </div>
            <div class="card">
                <div class="card-title">Open Positions</div>
                <div class="card-value neutral">{{ data.performance?.open_positions || 0 }}</div>
                <div class="card-subtitle">Max: 5</div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Kill Switch Monitor</div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <span :style="{ color: data.kill_switches?.kill_switch_active ? '#ff4444' : '#00ff88', fontWeight: 600 }">
                    {{ data.kill_switches?.kill_switch_active ? 'KILL SWITCH ACTIVE' : 'Kill Switch: OFF' }}
                </span>
                <button class="kill-btn" :class="{ deactivate: data.kill_switches?.kill_switch_active }"
                        @click="toggleKillSwitch">
                    {{ data.kill_switches?.kill_switch_active ? 'DEACTIVATE' : 'ACTIVATE KILL' }}
                </button>
            </div>
            <div class="kill-switch-grid">
                <div class="kill-layer" v-for="layer in data.kill_switches?.layers || []" :class="{ triggered: layer.triggered }">
                    <span class="kill-layer-name">{{ layer.name }}</span>
                    <span class="kill-layer-value">{{ layer.current }} / {{ layer.limit }}</span>
                </div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Market Context</div>
            <div class="market-indices">
                <div class="index-card">
                    <div class="index-name">VIX</div>
                    <div class="index-price" :style="{ color: getVixColor(data.market?.vix?.current) }">
                        {{ data.market?.vix?.current || 0 }}
                    </div>
                    <div style="font-size: 11px; color: #888;">{{ data.market?.vix?.regime || 'NORMAL' }}</div>
                </div>
                <div class="index-card">
                    <div class="index-name">SPY</div>
                    <div class="index-price">${{ data.market?.indices?.SPY || 0 }}</div>
                </div>
                <div class="index-card">
                    <div class="index-name">QQQ</div>
                    <div class="index-price">${{ data.market?.indices?.QQQ || 0 }}</div>
                </div>
                <div class="index-card">
                    <div class="index-name">IWM</div>
                    <div class="index-price">${{ data.market?.indices?.IWM || 0 }}</div>
                </div>
                <div class="index-card">
                    <div class="index-name">Market</div>
                    <div class="index-price" :style="{ color: data.market?.market?.is_open ? '#00ff88' : '#888' }">
                        {{ data.market?.market?.status || 'CLOSED' }}
                    </div>
                </div>
            </div>
        </div>

        <div class="section" v-if="data.positions && data.positions.length > 0">
            <div class="section-title">Open Positions</div>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Qty</th>
                        <th>Entry</th>
                        <th>Current</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="pos in data.positions">
                        <td style="font-weight: 600;">{{ pos.symbol }}</td>
                        <td :style="{ color: pos.side === 'long' ? '#00ff88' : '#ff4444' }">{{ pos.side.toUpperCase() }}</td>
                        <td>{{ pos.qty }}</td>
                        <td>${{ pos.entry_price.toFixed(2) }}</td>
                        <td>${{ pos.current_price.toFixed(2) }}</td>
                        <td :style="{ color: pos.unrealized_pnl >= 0 ? '#00ff88' : '#ff4444' }">
                            {{ formatPnl(pos.unrealized_pnl) }}
                        </td>
                        <td :style="{ color: pos.unrealized_pnl_pct >= 0 ? '#00ff88' : '#ff4444' }">
                            {{ pos.unrealized_pnl_pct.toFixed(2) }}%
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="section" v-if="data.signals && data.signals.length > 0">
            <div class="section-title">Recent Signals</div>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Direction</th>
                        <th>Entry</th>
                        <th>Stop</th>
                        <th>Target</th>
                        <th>R:R</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="sig in data.signals">
                        <td style="font-weight: 600;">{{ sig.symbol }}</td>
                        <td :style="{ color: sig.direction === 'long' ? '#00ff88' : '#ff4444' }">{{ sig.direction.toUpperCase() }}</td>
                        <td>${{ sig.entry_price }}</td>
                        <td>${{ sig.stop_loss }}</td>
                        <td>${{ sig.take_profit }}</td>
                        <td>{{ sig.rr_ratio }}:1</td>
                        <td>{{ sig.confidence }}%</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="footer">
            Kobe Trading System | Dashboard v1.0.0 | WebSocket: 5s refresh
        </div>
    </div>

    <script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
    <script>
        const { createApp, ref, onMounted, onUnmounted } = Vue;

        createApp({
            setup() {
                const data = ref({});
                const lastUpdate = ref('--:--:--');
                let ws = null;

                const formatNumber = (num) => {
                    return new Intl.NumberFormat('en-US', {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2
                    }).format(num);
                };

                const formatPnl = (pnl) => {
                    const sign = pnl >= 0 ? '+' : '';
                    return sign + '$' + formatNumber(Math.abs(pnl));
                };

                const getPnlClass = (pnl) => {
                    return pnl >= 0 ? 'positive' : 'negative';
                };

                const getVixColor = (vix) => {
                    if (vix < 15) return '#00ff88';
                    if (vix < 25) return '#ffaa00';
                    return '#ff4444';
                };

                const toggleKillSwitch = async () => {
                    const action = data.value.kill_switches?.kill_switch_active ? 'deactivate' : 'activate';
                    try {
                        await fetch(`/api/kill-switch/${action}`, { method: 'POST' });
                    } catch (e) {
                        console.error('Kill switch toggle failed:', e);
                    }
                };

                const connectWebSocket = () => {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(`${protocol}//${window.location.host}/ws/live`);

                    ws.onmessage = (event) => {
                        const msg = JSON.parse(event.data);
                        data.value = msg.data;
                        lastUpdate.value = new Date().toLocaleTimeString();
                    };

                    ws.onclose = () => {
                        setTimeout(connectWebSocket, 3000);
                    };

                    ws.onerror = (e) => {
                        console.error('WebSocket error:', e);
                    };
                };

                onMounted(() => {
                    // Initial data fetch
                    fetch('/api/data')
                        .then(r => r.json())
                        .then(d => {
                            data.value = d;
                            lastUpdate.value = new Date().toLocaleTimeString();
                        });

                    connectWebSocket();
                });

                onUnmounted(() => {
                    if (ws) ws.close();
                });

                return {
                    data,
                    lastUpdate,
                    formatNumber,
                    formatPnl,
                    getPnlClass,
                    getVixColor,
                    toggleKillSwitch
                };
            }
        }).mount('#app');
    </script>
</body>
</html>"""


# =============================================================================
# STARTUP FUNCTIONS
# =============================================================================

def start_dashboard(host: str = "127.0.0.1", port: int = 8080, background: bool = True):
    """
    Start the dashboard server.

    Args:
        host: Host to bind to
        port: Port to listen on
        background: Run in background thread

    Returns:
        Thread object if background=True, else None
    """
    print("=" * 60)
    print("KOBE TRADING DASHBOARD")
    print("Real-Time Monitoring")
    print("-" * 60)
    print(f"URL: http://{host}:{port}")
    print(f"WebSocket: ws://{host}:{port}/ws/live")
    print("Refresh Rate: 5 seconds")
    print("=" * 60)

    if background:
        thread = threading.Thread(
            target=uvicorn.run,
            kwargs={"app": app, "host": host, "port": port, "log_level": "warning"},
            daemon=True
        )
        thread.start()
        return thread
    else:
        uvicorn.run(app, host=host, port=port)
        return None


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kobe Trading Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    args = parser.parse_args()

    start_dashboard(host=args.host, port=args.port, background=False)
