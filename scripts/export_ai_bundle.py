#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), timeout=5)
        return out.decode().strip()
    except Exception:
        return None


def _sha256_file(p: Path) -> str | None:
    try:
        from core.config_pin import sha256_file  # type: ignore

        return sha256_file(str(p))
    except Exception:
        return None


def _coerce_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec: Dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, pd.Timestamp):
                rec[k] = v.isoformat()
            elif pd.isna(v):
                continue
            else:
                try:
                    rec[k] = v.item() if hasattr(v, "item") else v
                except Exception:
                    rec[k] = str(v)
        recs.append(rec)
    return recs


def main() -> int:
    ap = argparse.ArgumentParser(description="Export AI bundle with Top 3 and Trade of Day")
    ap.add_argument("--dotenv", type=str, default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env")
    ap.add_argument("--picks", type=str, default=str(ROOT / "logs" / "daily_picks.csv"))
    ap.add_argument("--totd", type=str, default=str(ROOT / "logs" / "trade_of_day.csv"))
    ap.add_argument("--out", type=str, default=str(ROOT / "logs" / "ai_bundle_latest.json"))
    args = ap.parse_args()

    picks_path = Path(args.picks)
    totd_path = Path(args.totd)

    if not picks_path.exists() or not totd_path.exists():
        print("Missing Top 3 or Trade of Day files. Run scan.py --top3 first.")
        return 2

    # Load CSVs
    picks_df = pd.read_csv(picks_path)
    totd_df = pd.read_csv(totd_path)

    # Build bundle
    commit = _git_commit()
    cfg_pin = _sha256_file(ROOT / "config" / "base.yaml")
    ts = datetime.utcnow().isoformat()
    day = datetime.utcnow().date().isoformat()

    bundle: Dict[str, Any] = {
        "generated_at": ts,
        "date": day,
        "universe_cap": 900,
        "git_commit": commit,
        "config_pin": cfg_pin,
        "top3": _coerce_records(picks_df),
        "trade_of_day": _coerce_records(totd_df)[0] if not totd_df.empty else {},
    }

    # Write latest
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    # Also write dated archive
    archive_dir = ROOT / "logs" / "ai_bundles"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_file = archive_dir / f"ai_bundle_{day}.json"
    archive_file.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path}")
    print(f"Wrote: {archive_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

