#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def build(final_path: Path, candidates_path: Path, out_path: Path, target: int = 900) -> int:
    if not final_path.exists() or not candidates_path.exists():
        raise FileNotFoundError('Missing source files')
    df_final = pd.read_csv(final_path)
    df_cand = pd.read_csv(candidates_path)
    col_final = 'symbol' if 'symbol' in df_final.columns else df_final.columns[0]
    col_cand = 'symbol' if 'symbol' in df_cand.columns else df_cand.columns[0]
    syms_final = df_final[col_final].astype(str).str.strip().str.upper()
    syms_cand = df_cand[col_cand].astype(str).str.strip().str.upper()
    uniq: list[str] = []
    seen: set[str] = set()
    for s in syms_final:
        if s and s not in seen:
            uniq.append(s); seen.add(s)
    for s in syms_cand:
        if len(uniq) >= target:
            break
        if s and s not in seen:
            uniq.append(s); seen.add(s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'symbol': uniq}).to_csv(out_path, index=False)
    return len(uniq)


def main() -> int:
    ap = argparse.ArgumentParser(description='Build 900-symbol universe from final + candidates')
    ap.add_argument('--final', type=str, default='data/universe/optionable_liquid_800.csv')
    ap.add_argument('--candidates', type=str, default='data/universe/optionable_liquid_candidates.csv')
    ap.add_argument('--out', type=str, default='data/universe/optionable_liquid_800.csv')
    ap.add_argument('--target', type=int, default=900)
    args = ap.parse_args()
    n = build(Path(args.final), Path(args.candidates), Path(args.out), target=int(args.target))
    print(f'Wrote {args.out} with {n} symbols')
    return 0 if n >= int(args.target) else 1


if __name__ == '__main__':
    raise SystemExit(main())
