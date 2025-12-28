#!/usr/bin/env python3
from __future__ import annotations

"""
Promote candidate models to deployed models if they improve metrics.

Reads candidates summary: state/models/candidates/meta_train_summary.json
Compares to deployed summary: state/models/deployed/meta_current.json
Promotion criteria (per strategy): accuracy >= current_accuracy + 0.01 and test_rows >= 100
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml_meta.model import CANDIDATE_DIR, DEPLOYED_DIR, model_paths
from monitor.drift_detector import DriftThresholds, compare_metrics
from core.alerts import send_telegram
from core.clock.tz_utils import fmt_ct, now_et
from core.journal import append_journal


def load_json(p: Path) -> Dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _backup_deployed(timestamp: str) -> Path:
    """Backup current deployed models into timestamped folder, return path."""
    bdir = DEPLOYED_DIR / 'backup' / timestamp
    bdir.mkdir(parents=True, exist_ok=True)
    for strat in ('ibs_rsi','turtle_soup'):
        pkl, meta = model_paths(strat, kind='deployed')
        if pkl.exists():
            (bdir / pkl.name).write_bytes(pkl.read_bytes())
        if meta.exists():
            (bdir / meta.name).write_text(meta.read_text(encoding='utf-8'), encoding='utf-8')
    # also copy meta_current.json if present
    cur = DEPLOYED_DIR / 'meta_current.json'
    if cur.exists():
        (bdir / 'meta_current.json').write_text(cur.read_text(encoding='utf-8'), encoding='utf-8')
    return bdir


def _restore_from_backup(backup_dir: Path) -> None:
    """Restore deployed models from a given backup directory."""
    for item in backup_dir.iterdir():
        if item.is_file():
            dst = DEPLOYED_DIR / item.name
            if item.suffix.lower() in ('.pkl', '.json') or item.name == 'meta_current.json':
                dst.parent.mkdir(parents=True, exist_ok=True)
                if item.suffix.lower() == '.pkl':
                    dst.write_bytes(item.read_bytes())
                else:
                    dst.write_text(item.read_text(encoding='utf-8'), encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='Promote candidate ML models if better')
    ap.add_argument('--min-delta', type=float, default=0.01, help='Min accuracy improvement to promote')
    ap.add_argument('--min-test', type=int, default=100, help='Minimum test rows required')
    ap.add_argument('--min-wr', type=float, default=0.5, help='Minimum win rate threshold on test')
    ap.add_argument('--min-pf', type=float, default=1.1, help='Minimum profit factor threshold on test')
    ap.add_argument('--drift-acc', type=float, default=-0.02, help='Max allowed delta in accuracy vs previous (negative)')
    ap.add_argument('--drift-wr', type=float, default=-0.02, help='Max allowed delta in win-rate vs previous (negative)')
    ap.add_argument('--drift-pf', type=float, default=-0.10, help='Max allowed delta in profit factor vs previous (negative)')
    ap.add_argument('--drift-sharpe', type=float, default=-0.10, help='Max allowed delta in sharpe vs previous (negative)')
    args = ap.parse_args()

    cand_summary = CANDIDATE_DIR / 'meta_train_summary.json'
    curr_summary = DEPLOYED_DIR / 'meta_current.json'
    cand = load_json(cand_summary)
    curr = load_json(curr_summary)

    promoted_any = False
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    # Prepare previous meta for drift comparison and backup before any promotion
    prev_meta_path = DEPLOYED_DIR / 'meta_prev.json'
    curr_meta_path = DEPLOYED_DIR / 'meta_current.json'
    prev_meta = load_json(prev_meta_path)
    curr_before = load_json(curr_meta_path)
    if curr_before:
        # Save a copy of current to prev
        DEPLOYED_DIR.mkdir(parents=True, exist_ok=True)
        prev_meta_path.write_text(json.dumps(curr_before, indent=2), encoding='utf-8')
        backup_dir = _backup_deployed(timestamp)
    else:
        backup_dir = None
    for strat in ('ibs_rsi','turtle_soup'):
        c = cand.get(strat, {}) if isinstance(cand, dict) else {}
        d = curr.get(strat, {}) if isinstance(curr, dict) else {}
        c_acc = float(c.get('accuracy', 0.0) or 0.0)
        c_rows = int(c.get('test_rows', 0) or 0)
        c_wr = float(c.get('wr', 0.0) or 0.0)
        c_pf = float(c.get('pf', 0.0) or 0.0)
        d_acc = float(d.get('accuracy', 0.0) or 0.0)
        # Gate: sample size, accuracy delta, and minimum WR/PF thresholds
        if c_rows >= args.min_test and c_acc >= d_acc + args.min_delta and c_wr >= args.min_wr and (c_pf >= args.min_pf or c_pf == float('inf')):
            # Promote: copy pkl/json from candidates to deployed
            src_pkl, src_meta = model_paths(strat, kind='candidate')
            dst_pkl, dst_meta = model_paths(strat, kind='deployed')
            DEPLOYED_DIR.mkdir(parents=True, exist_ok=True)
            if src_pkl.exists():
                shutil.copy2(src_pkl, dst_pkl)
            if src_meta.exists():
                shutil.copy2(src_meta, dst_meta)
            curr[strat] = c
            promoted_any = True
            print(f'Promoted {strat}: acc {d_acc:.3f} -> {c_acc:.3f} (rows={c_rows}, wr={c_wr:.2f}, pf={c_pf:.2f})')
            try:
                nowmsg = now_et(); stamp = f"{fmt_ct(nowmsg)} | {nowmsg.strftime('%I:%M %p').lstrip('0')} ET"
                send_telegram(f"<b>Model Promoted</b> {strat}: acc {d_acc:.3f} -> {c_acc:.3f} (rows={c_rows}, wr={c_wr:.2f}, pf={c_pf:.2f}) [{stamp}]")
            except Exception:
                pass
        else:
            print(f'Skipped {strat}: cand acc={c_acc:.3f}, rows={c_rows}, current acc={d_acc:.3f}')

    if promoted_any:
        DEPLOYED_DIR.mkdir(parents=True, exist_ok=True)
        curr_summary.write_text(json.dumps(curr, indent=2), encoding='utf-8')
        # Append audit record
        audit = DEPLOYED_DIR / 'meta_decisions.jsonl'
        audit.write_text('', encoding='utf-8') if not audit.exists() else None
        with open(audit, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'timestamp': datetime.utcnow().isoformat(),
                'promoted': curr,
                'criteria': {
                    'min_delta': args.min_delta,
                    'min_test': args.min_test,
                    'min_wr': args.min_wr,
                    'min_pf': args.min_pf,
                },
            }) + '\n')
        print('Updated deployed summary:', curr_summary)
        try:
            append_journal('model_promoted', {'summary': str(curr_summary)})
        except Exception:
            pass
    else:
        print('No promotions.')

    # Drift detection vs previous deployed meta; rollback if degraded beyond thresholds
    try:
        prev = load_json(prev_meta_path)
        cur = load_json(curr_meta_path)
        if prev and cur:
            thr = DriftThresholds(
                min_delta_accuracy=float(args.drift_acc),
                min_delta_wr=float(args.drift_wr),
                min_delta_pf=float(args.drift_pf),
                min_delta_sharpe=float(args.drift_sharpe),
            )
            rolled_back = []
            for strat in ('ibs_rsi','turtle_soup'):
                p = prev.get(strat, {}) if isinstance(prev, dict) else {}
                q = cur.get(strat, {}) if isinstance(cur, dict) else {}
                if not p or not q:
                    continue
                flags = compare_metrics(p, q, thr)
                if flags.get('any_drift'):
                    # Restore from backup if available
                    if backup_dir and backup_dir.exists():
                        _restore_from_backup(backup_dir)
                        rolled_back.append(strat)
            if rolled_back:
                msg = '<b>Model Rollback</b> due to drift: ' + ', '.join(rolled_back)
                print(msg)
                try:
                    nowrb = now_et(); stamprb = f"{fmt_ct(nowrb)} | {nowrb.strftime('%I:%M %p').lstrip('0')} ET"
                    send_telegram(f"{msg} [{stamprb}]")
                except Exception:
                    pass
                try:
                    append_journal('model_rollback', {'backup': str(backup_dir), 'strategies': rolled_back})
                except Exception:
                    pass
    except Exception as e:
        print('Drift/rollback stage error:', e)


if __name__ == '__main__':
    main()
