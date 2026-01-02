#!/usr/bin/env python3
"""
Full System Health Check for Kobe Trading Robot
Validates all components are working together correctly.
"""
from __future__ import annotations

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

def check_critical_files() -> Tuple[List[str], List[str]]:
    """Check all critical files exist."""
    ok = []
    issues = []
    critical_files = [
        'config/base.yaml',
        'config/settings.json',
        'scripts/scheduler_kobe.py',
        'scripts/scan.py',
        'scripts/run_paper_trade.py',
        'scripts/position_manager.py',
        'scripts/generate_briefing.py',
        'risk/weekly_exposure_gate.py',
        'risk/dynamic_position_sizer.py',
        'state/weekly_budget.json',
    ]
    for f in critical_files:
        if Path(f).exists():
            ok.append(f)
        else:
            issues.append(f'Missing: {f}')
    return ok, issues


def check_imports() -> Tuple[List[str], List[str]]:
    """Check all critical imports work."""
    ok = []
    issues = []

    imports = [
        ('strategies.dual_strategy', 'DualStrategyScanner'),
        ('risk.policy_gate', 'PolicyGate'),
        ('risk.weekly_exposure_gate', 'WeeklyExposureGate'),
        ('risk.dynamic_position_sizer', 'calculate_dynamic_allocations'),
        ('execution.broker_alpaca', 'place_bracket_order'),
        ('core.structured_log', 'jlog'),
    ]

    for mod, cls in imports:
        try:
            exec(f'from {mod} import {cls}')
            ok.append(f'{mod}.{cls}')
        except Exception as e:
            issues.append(f'Import {mod}.{cls}: {str(e)[:50]}')

    return ok, issues


def check_weekly_budget() -> Dict:
    """Check weekly budget state."""
    try:
        with open('state/weekly_budget.json') as f:
            budget = json.load(f)

        positions = budget.get('positions_opened', [])
        open_pos = [p for p in positions if p.get('status') == 'open']

        return {
            'ok': True,
            'week': f"{budget.get('week_start')} to {budget.get('week_end')}",
            'exposure': budget.get('current_exposure_pct', 0),
            'open_positions': len(open_pos),
            'positions': [p['symbol'] for p in open_pos],
        }
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def check_broker() -> Dict:
    """Check broker connection."""
    try:
        from execution.broker_alpaca import get_account_info
        info = get_account_info()
        if info:
            return {
                'ok': True,
                'equity': float(info.get('equity', 0)),
                'buying_power': float(info.get('buying_power', 0)),
            }
        return {'ok': False, 'error': 'No account info'}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def check_positions_broker() -> Dict:
    """Check positions from broker."""
    try:
        positions_file = Path('state/reconcile/positions.json')
        if positions_file.exists():
            with open(positions_file) as f:
                positions = json.load(f)
            return {
                'ok': True,
                'count': len(positions),
                'positions': [
                    {
                        'symbol': p['symbol'],
                        'qty': p['qty'],
                        'pnl': float(p.get('unrealized_pl', 0)),
                    }
                    for p in positions
                ],
            }
        return {'ok': False, 'error': 'No positions file'}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def check_reports() -> Dict:
    """Check reports generated today."""
    try:
        import pytz
        ET = pytz.timezone('America/New_York')
        today = datetime.now(ET).strftime('%Y%m%d')

        reports_dir = Path('reports')
        if not reports_dir.exists():
            return {'ok': False, 'error': 'Reports dir missing'}

        today_reports = list(reports_dir.glob(f'*{today}*'))
        return {
            'ok': True,
            'total': len(list(reports_dir.glob('*'))),
            'today': len(today_reports),
            'today_files': [r.name for r in today_reports],
        }
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def main():
    import pytz
    ET = pytz.timezone('America/New_York')
    now = datetime.now(ET)

    print('=' * 70)
    print('KOBE TRADING SYSTEM - FULL HEALTH CHECK')
    print(f'Timestamp: {now.strftime("%Y-%m-%d %H:%M:%S %Z")}')
    print('=' * 70)

    all_issues = []

    # 1. Critical Files
    print('\n[1] CRITICAL FILES')
    ok, issues = check_critical_files()
    print(f'    OK: {len(ok)}, Issues: {len(issues)}')
    for i in issues:
        print(f'    [X] {i}')
    all_issues.extend(issues)

    # 2. Imports
    print('\n[2] CORE IMPORTS')
    ok, issues = check_imports()
    print(f'    OK: {len(ok)}, Issues: {len(issues)}')
    for i in issues:
        print(f'    [X] {i}')
    all_issues.extend(issues)

    # 3. Weekly Budget
    print('\n[3] WEEKLY BUDGET')
    budget = check_weekly_budget()
    if budget['ok']:
        print(f'    Week: {budget["week"]}')
        print(f'    Exposure: {budget["exposure"]:.1%}')
        print(f'    Open Positions: {budget["positions"]}')
    else:
        print(f'    [X] Error: {budget["error"]}')
        all_issues.append(f'Weekly budget: {budget["error"]}')

    # 4. Broker
    print('\n[4] BROKER CONNECTION')
    broker = check_broker()
    if broker['ok']:
        print(f'    Equity: ${broker["equity"]:,.2f}')
        print(f'    Buying Power: ${broker["buying_power"]:,.2f}')
    else:
        print(f'    [X] Error: {broker["error"]}')
        all_issues.append(f'Broker: {broker["error"]}')

    # 5. Positions
    print('\n[5] BROKER POSITIONS')
    positions = check_positions_broker()
    if positions['ok']:
        print(f'    Count: {positions["count"]}')
        total_pnl = 0
        for p in positions['positions']:
            print(f'    - {p["symbol"]}: {p["qty"]} shares, P&L: ${p["pnl"]:+.2f}')
            total_pnl += p['pnl']
        print(f'    Total P&L: ${total_pnl:+.2f}')
    else:
        print(f'    [X] Error: {positions["error"]}')
        all_issues.append(f'Positions: {positions["error"]}')

    # 6. Reports
    print('\n[6] REPORTS')
    reports = check_reports()
    if reports['ok']:
        print(f'    Total reports: {reports["total"]}')
        print(f'    Today: {reports["today"]}')
        for r in reports['today_files'][:5]:
            print(f'    - {r}')
    else:
        print(f'    [X] Error: {reports["error"]}')
        all_issues.append(f'Reports: {reports["error"]}')

    # Summary
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)

    if all_issues:
        print(f'\n[!] {len(all_issues)} ISSUES FOUND:')
        for i, issue in enumerate(all_issues, 1):
            print(f'    {i}. {issue}')
        return 1
    else:
        print('\n[OK] ALL SYSTEMS HEALTHY!')
        return 0


if __name__ == '__main__':
    sys.exit(main())
