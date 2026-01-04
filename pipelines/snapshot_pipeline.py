"""
Snapshot Pipeline - Create immutable data snapshots.

This pipeline creates point-in-time snapshots of market data
for reproducible backtesting and audit trails.

Schedule: Nightly (22:00 ET)

Author: Kobe Trading System
Version: 1.0.0
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from pipelines.base import Pipeline


class SnapshotPipeline(Pipeline):
    """Pipeline for creating data snapshots."""

    @property
    def name(self) -> str:
        return "snapshot"

    def execute(self) -> bool:
        """
        Execute snapshot creation.

        Returns:
            True if snapshot created successfully
        """
        self.logger.info("Creating data snapshot...")

        # Create snapshot directory
        snapshot_date = datetime.utcnow().strftime("%Y%m%d")
        snapshot_dir = self.data_dir / "snapshots" / snapshot_date
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Create manifest
        manifest = {
            "snapshot_id": f"snapshot_{snapshot_date}",
            "created_at": datetime.utcnow().isoformat(),
            "universe_cap": self.universe_cap,
            "files": [],
        }

        # Copy cache files to snapshot
        cache_dir = self.data_dir / "cache" / "polygon_eod"
        if not cache_dir.exists():
            self.add_warning("No cache directory found")
            return True  # Not a failure, just nothing to snapshot

        symbols = self.load_universe()
        files_copied = 0

        for symbol in symbols[:self.universe_cap]:
            cache_file = cache_dir / f"{symbol}.csv"
            if cache_file.exists():
                # Calculate hash
                content = cache_file.read_bytes()
                file_hash = hashlib.sha256(content).hexdigest()

                # Copy file
                snapshot_file = snapshot_dir / f"{symbol}.csv"
                snapshot_file.write_bytes(content)

                manifest["files"].append({
                    "symbol": symbol,
                    "filename": f"{symbol}.csv",
                    "sha256": file_hash,
                    "size_bytes": len(content),
                })
                files_copied += 1

        # Save manifest
        manifest_file = snapshot_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2))

        self.set_metric("files_copied", files_copied)
        self.set_metric("snapshot_id", manifest["snapshot_id"])
        self.add_artifact(str(snapshot_dir))

        self.logger.info(f"Snapshot created: {files_copied} files")
        return True

    def get_latest_snapshot(self) -> Optional[Path]:
        """Get path to latest snapshot."""
        snapshots_dir = self.data_dir / "snapshots"
        if not snapshots_dir.exists():
            return None

        snapshots = sorted(snapshots_dir.iterdir(), reverse=True)
        return snapshots[0] if snapshots else None

    def verify_snapshot(self, snapshot_path: Path) -> bool:
        """Verify snapshot integrity using manifest hashes."""
        manifest_file = snapshot_path / "manifest.json"
        if not manifest_file.exists():
            return False

        manifest = json.loads(manifest_file.read_text())
        for file_info in manifest["files"]:
            file_path = snapshot_path / file_info["filename"]
            if not file_path.exists():
                self.add_error(f"Missing file: {file_info['filename']}")
                return False

            content = file_path.read_bytes()
            actual_hash = hashlib.sha256(content).hexdigest()
            if actual_hash != file_info["sha256"]:
                self.add_error(f"Hash mismatch: {file_info['filename']}")
                return False

        return True
