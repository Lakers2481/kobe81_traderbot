"""
Frozen Data Lake Manifest System
=================================

Provides immutable dataset semantics with cryptographic hashing.

Key Principles:
- Once a dataset_id exists, it NEVER changes
- dataset_id is derived from: provider, timeframe, start/end, universe hash, schema version
- Manifests contain per-file hashes for integrity verification
- KnowledgeBoundary can verify data hasn't drifted

Usage:
    from data.lake.manifest import DatasetManifest, compute_dataset_id

    manifest = DatasetManifest.create(
        provider='stooq',
        timeframe='1d',
        start_date='2015-01-01',
        end_date='2024-12-31',
        universe_path='data/universe/optionable_liquid_900.csv',
        schema_version='v1.0',
    )
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os

logger = logging.getLogger(__name__)

# Schema version - increment when data format changes
SCHEMA_VERSION = "v1.0"


@dataclass
class FileRecord:
    """Record of a single file in the dataset."""
    path: str
    sha256: str
    rows: int
    min_timestamp: str
    max_timestamp: str
    size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'FileRecord':
        return cls(**d)


@dataclass
class DatasetManifest:
    """
    Immutable manifest for a frozen dataset.

    Once created with a dataset_id, it should NEVER be modified.
    Any change to inputs creates a NEW dataset_id.
    """
    dataset_id: str
    provider: str
    timeframe: str
    start_date: str
    end_date: str
    universe_path: str
    universe_sha256: str
    schema_version: str
    created_at: str
    files: List[FileRecord] = field(default_factory=list)
    total_rows: int = 0
    total_symbols: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['files'] = [f.to_dict() for f in self.files]
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'DatasetManifest':
        files = [FileRecord.from_dict(f) for f in d.pop('files', [])]
        return cls(**d, files=files)

    def save(self, manifest_dir: Union[str, Path]) -> Path:
        """Save manifest to JSON file."""
        manifest_dir = Path(manifest_dir)
        manifest_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = manifest_dir / f"{self.dataset_id}.json"

        if manifest_path.exists():
            logger.warning(f"Manifest already exists: {manifest_path}")
            # Never overwrite - this is the immutability guarantee
            return manifest_path

        manifest_path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info(f"Saved manifest: {manifest_path}")
        return manifest_path

    @classmethod
    def load(cls, manifest_path: Union[str, Path]) -> 'DatasetManifest':
        """Load manifest from JSON file."""
        manifest_path = Path(manifest_path)
        data = json.loads(manifest_path.read_text())
        return cls.from_dict(data)

    def verify_integrity(self, lake_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Verify all files match their recorded hashes.

        Returns:
            Dict with 'valid', 'errors', 'warnings' keys
        """
        lake_dir = Path(lake_dir)
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'files_checked': 0,
            'files_missing': 0,
        }

        for file_record in self.files:
            file_path = lake_dir / file_record.path

            if not file_path.exists():
                result['valid'] = False
                result['errors'].append(f"Missing file: {file_record.path}")
                result['files_missing'] += 1
                continue

            # Compute current hash
            current_hash = compute_file_hash(file_path)

            if current_hash != file_record.sha256:
                result['valid'] = False
                result['errors'].append(
                    f"Hash mismatch for {file_record.path}: "
                    f"expected {file_record.sha256[:16]}..., got {current_hash[:16]}..."
                )
            else:
                result['files_checked'] += 1

        return result

    def get_coverage_stats(self) -> Dict[str, Any]:
        """Get dataset coverage statistics."""
        if not self.files:
            return {'years': 0, 'symbols': 0, 'total_rows': 0}

        try:
            start = datetime.fromisoformat(self.start_date)
            end = datetime.fromisoformat(self.end_date)
            years = (end - start).days / 365.25
        except:
            years = 0

        return {
            'years': years,
            'symbols': self.total_symbols,
            'total_rows': self.total_rows,
            'file_count': len(self.files),
        }


def compute_file_hash(path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """Compute cryptographic hash of a file."""
    path = Path(path)
    hasher = hashlib.new(algorithm)

    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def compute_dataset_id(
    provider: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    universe_path: Union[str, Path],
    schema_version: str = SCHEMA_VERSION,
) -> str:
    """
    Compute deterministic dataset ID from inputs.

    This ensures:
    - Same inputs always produce same dataset_id
    - Different inputs always produce different dataset_id
    - dataset_id is unique fingerprint of the frozen data
    """
    universe_path = Path(universe_path)

    # Get universe file hash
    if universe_path.exists():
        universe_hash = compute_file_hash(universe_path)
    else:
        universe_hash = "empty"

    # Build input string
    input_str = "|".join([
        provider,
        timeframe,
        start_date,
        end_date,
        universe_hash,
        schema_version,
    ])

    # Hash to create dataset_id
    full_hash = hashlib.sha256(input_str.encode()).hexdigest()

    # Use readable prefix + truncated hash
    dataset_id = f"{provider}_{timeframe}_{start_date[:4]}_{end_date[:4]}_{full_hash[:12]}"

    return dataset_id


def create_manifest(
    provider: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    universe_path: Union[str, Path],
    data_files: List[Path],
    schema_version: str = SCHEMA_VERSION,
    metadata: Optional[Dict] = None,
) -> DatasetManifest:
    """
    Create a new dataset manifest from data files.

    Args:
        provider: Data provider name (e.g., 'stooq', 'binance')
        timeframe: Data timeframe (e.g., '1d', '1h')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        universe_path: Path to universe CSV
        data_files: List of data file paths
        schema_version: Schema version string
        metadata: Optional additional metadata

    Returns:
        DatasetManifest object
    """
    universe_path = Path(universe_path)

    # Compute dataset ID
    dataset_id = compute_dataset_id(
        provider, timeframe, start_date, end_date, universe_path, schema_version
    )

    # Compute universe hash
    universe_hash = compute_file_hash(universe_path) if universe_path.exists() else "empty"

    # Build file records
    file_records = []
    total_rows = 0
    symbols = set()

    for file_path in data_files:
        file_path = Path(file_path)
        if not file_path.exists():
            continue

        # Get file info
        file_hash = compute_file_hash(file_path)
        file_size = file_path.stat().st_size

        # Try to read and get row count + timestamps
        rows = 0
        min_ts = ""
        max_ts = ""

        try:
            if file_path.suffix == '.parquet':
                import pandas as pd
                df = pd.read_parquet(file_path)
            elif file_path.suffix == '.csv':
                import pandas as pd
                df = pd.read_csv(file_path)
            else:
                df = None

            if df is not None:
                rows = len(df)
                total_rows += rows

                if 'symbol' in df.columns:
                    symbols.update(df['symbol'].unique())

                if 'timestamp' in df.columns:
                    min_ts = str(df['timestamp'].min())
                    max_ts = str(df['timestamp'].max())
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")

        file_records.append(FileRecord(
            path=str(file_path.name),
            sha256=file_hash,
            rows=rows,
            min_timestamp=min_ts,
            max_timestamp=max_ts,
            size_bytes=file_size,
        ))

    return DatasetManifest(
        dataset_id=dataset_id,
        provider=provider,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        universe_path=str(universe_path),
        universe_sha256=universe_hash,
        schema_version=schema_version,
        created_at=datetime.now().isoformat(),
        files=file_records,
        total_rows=total_rows,
        total_symbols=len(symbols),
        metadata=metadata or {},
    )


def find_manifest(
    dataset_id: str,
    manifest_dir: Union[str, Path] = "data/manifests",
) -> Optional[DatasetManifest]:
    """Find and load a manifest by dataset_id."""
    manifest_dir = Path(manifest_dir)
    manifest_path = manifest_dir / f"{dataset_id}.json"

    if manifest_path.exists():
        return DatasetManifest.load(manifest_path)

    return None


def list_manifests(
    manifest_dir: Union[str, Path] = "data/manifests",
) -> List[DatasetManifest]:
    """List all available manifests."""
    manifest_dir = Path(manifest_dir)
    manifests = []

    if not manifest_dir.exists():
        return manifests

    for manifest_path in manifest_dir.glob("*.json"):
        try:
            manifests.append(DatasetManifest.load(manifest_path))
        except Exception as e:
            logger.warning(f"Could not load manifest {manifest_path}: {e}")

    return sorted(manifests, key=lambda m: m.created_at, reverse=True)


def dataset_exists(dataset_id: str, manifest_dir: Union[str, Path] = "data/manifests") -> bool:
    """Check if a dataset already exists."""
    return find_manifest(dataset_id, manifest_dir) is not None
