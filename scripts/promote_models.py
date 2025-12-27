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
from core.journal import append_journal


def load_json(p: Path) -> Dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description='Promote candidate ML models if better')
    ap.add_argument('--min-delta', type=float, default=0.01, help='Min accuracy improvement to promote')
    ap.add_argument('--min-test', type=int, default=100, help='Minimum test rows required')
    ap.add_argument('--min-wr', type=float, default=0.5, help='Minimum win rate threshold on test')
    ap.add_argument('--min-pf', type=float, default=1.1, help='Minimum profit factor threshold on test')
    args = ap.parse_args()

    cand_summary = CANDIDATE_DIR / 'meta_train_summary.json'
    curr_summary = DEPLOYED_DIR / 'meta_current.json'
    cand = load_json(cand_summary)
    curr = load_json(curr_summary)

    promoted_any = False
    for strat in ('donchian','turtle_soup'):
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


if __name__ == '__main__':
    main()
