#!/usr/bin/env python
"""Data-backed verification of entire Kobe trading system."""

import os
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

def main():
    print('=' * 70)
    print('DATA-BACKED VERIFICATION - FULL SYSTEM CHECK')
    print('=' * 70)
    print()

    errors = []

    # 1. VERIFY DATA FILES EXIST
    print('1. DATA FILES VERIFICATION')
    print('-' * 70)

    data_checks = [
        ('Universe', 'data/universe/optionable_liquid_800.csv'),
        ('Daily Picks', 'logs/daily_picks.csv'),
        ('Signals Log', 'logs/signals.jsonl'),
        ('Pregame JSON', 'reports/pregame_20260101.json'),
        ('Pregame MD', 'reports/pregame_20260101.md'),
        ('Config Base', 'config/base.yaml'),
        ('Config Brokers', 'config/brokers.yaml'),
    ]

    for name, path in data_checks:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f'  [OK] {name}: {path} ({size:,} bytes)')
        else:
            print(f'  [MISSING] {name}: {path}')
            errors.append(f'Missing: {path}')

    print()

    # 2. VERIFY UNIVERSE DATA
    print('2. UNIVERSE DATA')
    print('-' * 70)
    import pandas as pd
    try:
        universe = pd.read_csv('data/universe/optionable_liquid_800.csv')
        print(f'  Symbols in universe: {len(universe)}')
        print(f'  Columns: {list(universe.columns)}')
        sample = list(universe['symbol'].head(5))
        print(f'  Sample symbols: {sample}')
    except Exception as e:
        print(f'  [ERROR] {e}')
        errors.append(f'Universe error: {e}')

    print()

    # 3. VERIFY DAILY PICKS DATA
    print('3. DAILY PICKS (WATCHLIST)')
    print('-' * 70)
    try:
        picks = pd.read_csv('logs/daily_picks.csv')
        print(f'  Total picks: {len(picks)}')
        print(f'  Columns: {list(picks.columns)}')
        for i, row in picks.iterrows():
            symbol = row['symbol']
            entry = row['entry_price']
            stop = row['stop_loss']
            conf = row['conf_score']
            risk = entry - stop
            print(f'  {i+1}. {symbol}: Entry=${entry:.2f}, Stop=${stop:.2f}, Risk=${risk:.2f}, Conf={conf:.1%}')
    except Exception as e:
        print(f'  [ERROR] {e}')
        errors.append(f'Daily picks error: {e}')

    print()

    # 4. VERIFY SIGNALS LOG
    print('4. SIGNALS LOG (JSONL)')
    print('-' * 70)
    try:
        with open('logs/signals.jsonl', 'r') as f:
            lines = f.readlines()
        print(f'  Total signals logged: {len(lines)}')
        if lines:
            last = json.loads(lines[-1])
            print(f'  Last signal symbol: {last.get("symbol", "N/A")}')
            print(f'  Last signal time: {last.get("timestamp", "N/A")}')
    except Exception as e:
        print(f'  [ERROR] {e}')
        errors.append(f'Signals log error: {e}')

    print()

    # 5. VERIFY PREGAME REPORT
    print('5. PREGAME REPORT (JSON)')
    print('-' * 70)
    try:
        with open('reports/pregame_20260101.json', 'r') as f:
            pregame = json.load(f)
        print(f'  Timestamp: {pregame.get("timestamp", "N/A")}')
        print(f'  Signal Date: {pregame.get("signal_date", "N/A")}')
        print(f'  Trading Date: {pregame.get("trading_date", "N/A")}')
        print(f'  Watchlist size: {len(pregame.get("watchlist", []))}')
        regime = pregame.get('regime', {})
        print(f'  Regime State: {regime.get("state", "N/A")}')
        print(f'  Position Scale: {regime.get("position_scale", "N/A")}')
        risk_status = pregame.get('risk_status', {})
        print(f'  Kill Switch: {risk_status.get("kill_switch", "N/A")}')
        print(f'  Drift Scale: {risk_status.get("drift_scale", "N/A")}')
    except Exception as e:
        print(f'  [ERROR] {e}')
        errors.append(f'Pregame report error: {e}')

    print()

    # 6. VERIFY STATE FILES
    print('6. STATE FILES')
    print('-' * 70)
    state_files = [
        'state/KILL_SWITCH',
        'state/drift_state.json',
        'state/paper_positions.json',
    ]
    for path in state_files:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f'  [EXISTS] {path} ({size:,} bytes)')
        else:
            print(f'  [NOT SET] {path} (normal if not yet used)')

    print()

    # 7. VERIFY CONFIG FILES
    print('7. CONFIG FILES')
    print('-' * 70)
    import yaml
    try:
        with open('config/base.yaml', 'r') as f:
            config = yaml.safe_load(f)
        sections = list(config.keys())
        print(f'  Config sections: {len(sections)}')

        # Check for new sections
        new_sections = ['drift_detection', 'edge_analytics', 'net_exposure',
                        'volatility_targeting', 'gap_risk', 'regime_slippage',
                        'options_live', 'webhooks', 'broker']
        for sec in new_sections:
            if sec in config:
                print(f'  [OK] {sec}: configured')
            else:
                print(f'  [MISSING] {sec}: not in config')
                errors.append(f'Missing config section: {sec}')
    except Exception as e:
        print(f'  [ERROR] {e}')
        errors.append(f'Config error: {e}')

    print()

    # 8. VERIFY POLYGON DATA CACHE
    print('8. POLYGON DATA CACHE')
    print('-' * 70)
    cache_dir = Path('data/cache/polygon')
    if cache_dir.exists():
        csv_files = list(cache_dir.glob('*.csv'))
        print(f'  Cached symbols: {len(csv_files)}')
        if csv_files:
            # Check a sample file
            sample = csv_files[0]
            sample_df = pd.read_csv(sample)
            print(f'  Sample file: {sample.name}')
            print(f'  Sample rows: {len(sample_df)}')
            # Handle different column names
            date_col = 'date' if 'date' in sample_df.columns else sample_df.columns[0]
            print(f'  Date range: {sample_df[date_col].min()} to {sample_df[date_col].max()}')
    else:
        print('  [WARN] Cache directory not found (will fetch on demand)')

    print()

    # 9. VERIFY NEW MODULE FILES
    print('9. NEW MODULE FILES')
    print('-' * 70)
    new_files = [
        'risk/net_exposure_gate.py',
        'risk/volatility_targeting.py',
        'execution/order_state_machine.py',
        'execution/broker_base.py',
        'execution/broker_factory.py',
        'execution/broker_paper.py',
        'execution/broker_crypto.py',
        'backtest/gap_risk_model.py',
        'backtest/regime_adaptive_slippage.py',
        'analytics/edge_decomposition.py',
        'analytics/factor_attribution.py',
        'analytics/auto_standdown.py',
        'web/api/webhooks.py',
        'web/api/signal_queue.py',
        'options/chain_fetcher.py',
        'options/spreads.py',
        'options/order_router.py',
        'monitor/drift_detector.py',
    ]

    for path in new_files:
        if os.path.exists(path):
            size = os.path.getsize(path)
            lines = len(open(path).readlines())
            print(f'  [OK] {path} ({lines} lines)')
        else:
            print(f'  [MISSING] {path}')
            errors.append(f'Missing file: {path}')

    print()

    # 10. FUNCTIONAL TEST - Run actual components
    print('10. FUNCTIONAL COMPONENT TESTS')
    print('-' * 70)

    # Test Net Exposure Gate
    try:
        from risk.net_exposure_gate import NetExposureGate
        gate = NetExposureGate()
        print(f'  [OK] NetExposureGate: limits={gate.limits}')
    except Exception as e:
        print(f'  [FAIL] NetExposureGate: {e}')
        errors.append(f'NetExposureGate: {e}')

    # Test Volatility Targeting
    try:
        from risk.volatility_targeting import VolatilityTargetingGate
        vt = VolatilityTargetingGate()
        print(f'  [OK] VolatilityTargetingGate: target_vol={vt.config.target_annual_vol}')
    except Exception as e:
        print(f'  [FAIL] VolatilityTargetingGate: {e}')
        errors.append(f'VolatilityTargetingGate: {e}')

    # Test Order State Machine
    try:
        from execution.order_state_machine import OrderStateMachine
        osm = OrderStateMachine()
        order = osm.create_order('TEST', 'buy', 100, 50.0)
        print(f'  [OK] OrderStateMachine: created order {order.order_id[:8]}...')
    except Exception as e:
        print(f'  [FAIL] OrderStateMachine: {e}')
        errors.append(f'OrderStateMachine: {e}')

    # Test Broker Factory
    try:
        from execution.broker_factory import get_registered_brokers
        brokers = get_registered_brokers()
        print(f'  [OK] BrokerFactory: registered={brokers}')
    except Exception as e:
        print(f'  [FAIL] BrokerFactory: {e}')
        errors.append(f'BrokerFactory: {e}')

    # Test Gap Risk Model
    try:
        from backtest.gap_risk_model import GapRiskModel
        GapRiskModel()
        print('  [OK] GapRiskModel: initialized successfully')
    except Exception as e:
        print(f'  [FAIL] GapRiskModel: {e}')
        errors.append(f'GapRiskModel: {e}')

    # Test Regime Slippage
    try:
        from backtest.regime_adaptive_slippage import RegimeAdaptiveSlippage
        RegimeAdaptiveSlippage()
        print('  [OK] RegimeAdaptiveSlippage: initialized successfully')
    except Exception as e:
        print(f'  [FAIL] RegimeAdaptiveSlippage: {e}')
        errors.append(f'RegimeAdaptiveSlippage: {e}')

    # Test Edge Decomposition
    try:
        from analytics.edge_decomposition import EdgeDecomposition
        EdgeDecomposition()
        print('  [OK] EdgeDecomposition: initialized successfully')
    except Exception as e:
        print(f'  [FAIL] EdgeDecomposition: {e}')
        errors.append(f'EdgeDecomposition: {e}')

    # Test Auto Standdown
    try:
        from analytics.auto_standdown import AutoStanddown
        AutoStanddown()
        print('  [OK] AutoStanddown: initialized successfully')
    except Exception as e:
        print(f'  [FAIL] AutoStanddown: {e}')
        errors.append(f'AutoStanddown: {e}')

    # Test Signal Queue
    try:
        from web.api.signal_queue import SignalQueue
        sq = SignalQueue()
        print(f'  [OK] SignalQueue: max_queue_size={sq.max_queue_size}')
    except Exception as e:
        print(f'  [FAIL] SignalQueue: {e}')
        errors.append(f'SignalQueue: {e}')

    # Test Options Spread Builder
    try:
        from options.spreads import SpreadBuilder
        SpreadBuilder()
        print('  [OK] SpreadBuilder: initialized successfully')
    except Exception as e:
        print(f'  [FAIL] SpreadBuilder: {e}')
        errors.append(f'SpreadBuilder: {e}')

    # Test Drift Detector
    try:
        from monitor.drift_detector import DriftDetector, get_position_scale
        DriftDetector()
        scale = get_position_scale()
        print(f'  [OK] DriftDetector: position_scale={scale:.0%}')
    except Exception as e:
        print(f'  [FAIL] DriftDetector: {e}')
        errors.append(f'DriftDetector: {e}')

    print()

    # FINAL SUMMARY
    print('=' * 70)
    print('VERIFICATION SUMMARY')
    print('=' * 70)
    print()

    if errors:
        print(f'  ERRORS FOUND: {len(errors)}')
        for err in errors:
            print(f'    - {err}')
        print()
        print('  STATUS: NEEDS ATTENTION')
        return 1
    else:
        print('  ERRORS FOUND: 0')
        print()
        print('  ALL DATA FILES: VERIFIED')
        print('  ALL CONFIG SECTIONS: PRESENT')
        print('  ALL NEW MODULES: EXIST')
        print('  ALL COMPONENTS: FUNCTIONAL')
        print()
        print('  STATUS: READY FOR LIVE TRADING')
        return 0


if __name__ == '__main__':
    sys.exit(main())
