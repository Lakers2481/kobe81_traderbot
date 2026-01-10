"""
Frozen Data Lake Module
========================

Provides immutable dataset storage and management for reproducible backtesting.

Key Components:
- DatasetManifest: Immutable manifest with cryptographic hashes
- LakeWriter: Freeze DataFrames into the lake
- LakeReader: Load frozen datasets with integrity verification

Design Principles:
1. IMMUTABILITY: Once written, data never changes
2. INTEGRITY: Cryptographic hashes verify data hasn't drifted
3. REPRODUCIBILITY: Same dataset_id always returns same data
4. VERSIONING: Schema versions track format changes

Usage:
    from data.lake import LakeWriter, LakeReader, quick_load, quick_freeze

    # Freeze a dataset
    writer = LakeWriter()
    manifest = writer.freeze_dataframe(
        df=ohlcv_data,
        provider='stooq',
        timeframe='1d',
        universe_path='data/universe/optionable_liquid_800.csv',
    )

    # Load frozen data
    reader = LakeReader()
    df = reader.load_dataset(dataset_id=manifest.dataset_id)

    # Quick helpers
    df = quick_load('stooq_1d_2015_2024_abc123')
"""
from .manifest import (
    DatasetManifest,
    FileRecord,
    compute_dataset_id,
    compute_file_hash,
    create_manifest,
    find_manifest,
    list_manifests,
    dataset_exists,
    SCHEMA_VERSION,
)

from .io import (
    LakeWriter,
    LakeReader,
    quick_load,
    quick_freeze,
)

__all__ = [
    # Manifest
    'DatasetManifest',
    'FileRecord',
    'compute_dataset_id',
    'compute_file_hash',
    'create_manifest',
    'find_manifest',
    'list_manifests',
    'dataset_exists',
    'SCHEMA_VERSION',
    # I/O
    'LakeWriter',
    'LakeReader',
    'quick_load',
    'quick_freeze',
]
