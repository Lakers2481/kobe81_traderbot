#!/usr/bin/env python3
"""
Web dashboard for Kobe trading system.
Provides real-time trading status, positions, and performance.
Usage: python scripts/dashboard.py [--start|--stop|--status] [--port PORT]
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Dict, Optional

# Dashboard configuration
DEFAULT_PORT = 8080
PID_FILE = Path("state/dashboard.pid")


class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom handler for dashboard requests."""

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/":
            self.send_dashboard_html()
        elif self.path == "/api/status":
            self.send_json_response(get_system_status())
        elif self.path == "/api/positions":
            self.send_json_response(get_positions())
        elif self.path == "/api/pnl":
            self.send_json_response(get_pnl_summary())
        elif self.path == "/api/orders":
            self.send_json_response(get_recent_orders())
        elif self.path == "/api/health":
            self.send_json_response({"status": "healthy", "timestamp": datetime.now().isoformat()})
        else:
            super().do_GET()

    def send_json_response(self, data: Dict):
        """Send JSON response."""
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode())

    def send_dashboard_html(self):
        """Send dashboard HTML page."""
        html = get_dashboard_html()
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def get_system_status() -> Dict:
    """Get current system status."""
    status = {
        "timestamp": datetime.now().isoformat(),
        "system": "Kobe Trading System",
        "running": True,
        "kill_switch": Path("state/KILL_SWITCH").exists(),
        "mode": "unknown",
    }

    # Check mode
    alpaca_url = os.environ.get("ALPACA_BASE_URL", "")
    if "paper" in alpaca_url.lower():
        status["mode"] = "paper"
    elif "api.alpaca.markets" in alpaca_url:
        status["mode"] = "live"

    # Check runner status
    runner_state = Path("state/runner_state.json")
    if runner_state.exists():
        try:
            with open(runner_state) as f:
                state = json.load(f)
                status["runner"] = state
        except (json.JSONDecodeError, OSError):
            pass

    return status


def get_positions() -> Dict:
    """Get current positions."""
    positions_file = Path("state/positions.json")
    if not positions_file.exists():
        return {"positions": [], "count": 0, "total_value": 0}

    try:
        with open(positions_file) as f:
            positions = json.load(f)

        if isinstance(positions, dict):
            positions = list(positions.values())

        total_value = sum(
            float(p.get("qty", p.get("quantity", 0))) *
            float(p.get("current_price", p.get("avg_entry_price", 0)))
            for p in positions
        )

        return {
            "positions": positions,
            "count": len(positions),
            "total_value": total_value,
        }
    except Exception as e:
        return {"positions": [], "count": 0, "error": str(e)}


def get_pnl_summary() -> Dict:
    """Get P&L summary."""
    pnl_file = Path("state/pnl_summary.json")
    if pnl_file.exists():
        try:
            with open(pnl_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # Calculate from trades if available
    trade_log = Path("logs/trades.jsonl")
    if trade_log.exists():
        total_pnl = 0
        trade_count = 0
        wins = 0

        try:
            with open(trade_log) as f:
                for line in f:
                    trade = json.loads(line.strip())
                    pnl = float(trade.get("pnl", trade.get("realized_pnl", 0)))
                    total_pnl += pnl
                    trade_count += 1
                    if pnl > 0:
                        wins += 1

            return {
                "total_pnl": total_pnl,
                "trade_count": trade_count,
                "win_rate": (wins / trade_count * 100) if trade_count > 0 else 0,
            }
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    return {"total_pnl": 0, "trade_count": 0, "win_rate": 0}


def get_recent_orders() -> Dict:
    """Get recent orders."""
    orders_file = Path("state/order_history.json")
    if not orders_file.exists():
        return {"orders": [], "count": 0}

    try:
        with open(orders_file) as f:
            orders = json.load(f)

        if isinstance(orders, dict):
            orders = list(orders.values())

        # Get last 20 orders
        orders = sorted(orders, key=lambda x: x.get("timestamp", ""), reverse=True)[:20]

        return {"orders": orders, "count": len(orders)}
    except Exception as e:
        return {"orders": [], "count": 0, "error": str(e)}


def get_dashboard_html() -> str:
    """Generate dashboard HTML."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kobe Trading Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 15px;
            background: #16213e;
            border-radius: 8px;
        }
        .header h1 { font-size: 24px; color: #00d4ff; }
        .status-badge {
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
        }
        .status-running { background: #00c853; color: #000; }
        .status-stopped { background: #ff5252; color: #fff; }
        .status-paper { background: #ffc107; color: #000; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .card {
            background: #16213e;
            border-radius: 8px;
            padding: 20px;
        }
        .card h2 {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 15px;
        }
        .metric {
            font-size: 32px;
            font-weight: bold;
            color: #00d4ff;
        }
        .metric.positive { color: #00c853; }
        .metric.negative { color: #ff5252; }
        .positions-table {
            width: 100%;
            border-collapse: collapse;
        }
        .positions-table th, .positions-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #333;
        }
        .positions-table th { color: #888; font-weight: normal; }
        .refresh-time { font-size: 12px; color: #666; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏀 Kobe Trading Dashboard</h1>
        <div>
            <span id="mode-badge" class="status-badge">Loading...</span>
            <span id="status-badge" class="status-badge">Loading...</span>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>Total P&L</h2>
            <div id="total-pnl" class="metric">$0.00</div>
        </div>
        <div class="card">
            <h2>Win Rate</h2>
            <div id="win-rate" class="metric">0%</div>
        </div>
        <div class="card">
            <h2>Total Trades</h2>
            <div id="trade-count" class="metric">0</div>
        </div>
        <div class="card">
            <h2>Open Positions</h2>
            <div id="position-count" class="metric">0</div>
        </div>
    </div>

    <div class="card" style="margin-top: 20px;">
        <h2>Open Positions</h2>
        <table class="positions-table">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Qty</th>
                    <th>Entry</th>
                    <th>Current</th>
                    <th>P&L</th>
                </tr>
            </thead>
            <tbody id="positions-body">
                <tr><td colspan="5">Loading...</td></tr>
            </tbody>
        </table>
    </div>

    <div class="refresh-time">Last updated: <span id="last-update">Never</span></div>

    <script>
        async function fetchData() {
            try {
                const [status, positions, pnl] = await Promise.all([
                    fetch('/api/status').then(r => r.json()),
                    fetch('/api/positions').then(r => r.json()),
                    fetch('/api/pnl').then(r => r.json())
                ]);

                // Update status badges
                const modeBadge = document.getElementById('mode-badge');
                modeBadge.textContent = status.mode.toUpperCase();
                modeBadge.className = 'status-badge status-' + status.mode;

                const statusBadge = document.getElementById('status-badge');
                if (status.kill_switch) {
                    statusBadge.textContent = 'STOPPED';
                    statusBadge.className = 'status-badge status-stopped';
                } else {
                    statusBadge.textContent = 'RUNNING';
                    statusBadge.className = 'status-badge status-running';
                }

                // Update metrics
                const totalPnl = pnl.total_pnl || 0;
                const pnlEl = document.getElementById('total-pnl');
                pnlEl.textContent = '$' + totalPnl.toFixed(2);
                pnlEl.className = 'metric ' + (totalPnl >= 0 ? 'positive' : 'negative');

                document.getElementById('win-rate').textContent = (pnl.win_rate || 0).toFixed(1) + '%';
                document.getElementById('trade-count').textContent = pnl.trade_count || 0;
                document.getElementById('position-count').textContent = positions.count || 0;

                // Update positions table
                const tbody = document.getElementById('positions-body');
                if (positions.positions && positions.positions.length > 0) {
                    tbody.innerHTML = positions.positions.map(p => {
                        const qty = p.qty || p.quantity || 0;
                        const entry = p.avg_entry_price || 0;
                        const current = p.current_price || entry;
                        const pnl = (current - entry) * qty;
                        return `<tr>
                            <td>${p.symbol}</td>
                            <td>${qty}</td>
                            <td>$${entry.toFixed(2)}</td>
                            <td>$${current.toFixed(2)}</td>
                            <td class="${pnl >= 0 ? 'positive' : 'negative'}">$${pnl.toFixed(2)}</td>
                        </tr>`;
                    }).join('');
                } else {
                    tbody.innerHTML = '<tr><td colspan="5">No open positions</td></tr>';
                }

                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
            } catch (e) {
                console.error('Error fetching data:', e);
            }
        }

        fetchData();
        setInterval(fetchData, 5000);
    </script>
</body>
</html>"""


def start_dashboard(port: int):
    """Start the dashboard server."""
    # Check if already running
    if PID_FILE.exists():
        with open(PID_FILE) as f:
            old_pid = f.read().strip()
        print(f"Dashboard may already be running (PID: {old_pid})")
        print(f"Use --stop to stop it first, or --status to check")
        return

    print(f"\n=== Starting Kobe Dashboard ===\n")
    print(f"Starting server on http://localhost:{port}")

    # Save PID
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    try:
        server = HTTPServer(("", port), DashboardHandler)
        print(f"Dashboard running at http://localhost:{port}")
        print("Press Ctrl+C to stop\n")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
    finally:
        if PID_FILE.exists():
            PID_FILE.unlink()


def stop_dashboard():
    """Stop the dashboard server."""
    if not PID_FILE.exists():
        print("Dashboard is not running")
        return

    with open(PID_FILE) as f:
        pid = int(f.read().strip())

    print(f"Stopping dashboard (PID: {pid})...")

    try:
        os.kill(pid, signal.SIGTERM)
        print("Dashboard stopped")
    except ProcessLookupError:
        print("Dashboard process not found (may have already stopped)")
    except Exception as e:
        print(f"Error stopping dashboard: {e}")

    if PID_FILE.exists():
        PID_FILE.unlink()


def show_status():
    """Show dashboard status."""
    print("\n=== Dashboard Status ===\n")

    if PID_FILE.exists():
        with open(PID_FILE) as f:
            pid = f.read().strip()
        print(f"Status: RUNNING")
        print(f"PID: {pid}")

        # Try to check if process is alive
        try:
            os.kill(int(pid), 0)
            print("Process: Active")
        except ProcessLookupError:
            print("Process: Dead (stale PID file)")
            PID_FILE.unlink()
    else:
        print("Status: STOPPED")

    print("\nTo start: python scripts/dashboard.py --start")
    print("To stop:  python scripts/dashboard.py --stop")


def main():
    parser = argparse.ArgumentParser(description="Kobe trading dashboard")
    parser.add_argument("--start", action="store_true", help="Start the dashboard")
    parser.add_argument("--stop", action="store_true", help="Stop the dashboard")
    parser.add_argument("--status", action="store_true", help="Show dashboard status")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port number (default: {DEFAULT_PORT})")

    args = parser.parse_args()

    if args.stop:
        stop_dashboard()
    elif args.status:
        show_status()
    elif args.start:
        start_dashboard(args.port)
    else:
        # Default: show status
        show_status()


if __name__ == "__main__":
    main()
