"""
Frozen Data Lake I/O Module
============================

Handles reading and writing immutable datasets to the data lake.

Key Principles:
- Write once, never modify (immutability)
- Parquet preferred for performance, CSV fallback
- All writes go through manifest system
- Reads verify integrity when requested

Usage:
    from data.lake.io import LakeWriter, LakeReader

    # Write frozen data
    writer = LakeWriter(lake_dir='data/lake')
    manifest = writer.freeze_dataframe(
        df=ohlcv_data,
        provider='stooq',
        timeframe='1d',
        universe_path='data/universe/optionable_liquid_final.csv',
    )

    # Read frozen data
    reader = LakeReader(lake_dir='data/lake')
    df = reader.load_dataset(dataset_id='stooq_1d_2015_2024_abc123')
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

import pandas as pd

from .manifest import (
    DatasetManifest,
    compute_dataset_id,
    create_manifest,
    find_manifest,
    dataset_exists,
    compute_file_hash,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)

# Try to import pyarrow for parquet support
try:
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    logger.warning("pyarrow not available, using CSV fallback")


class LakeWriter:
    """
    Writes immutable datasets to the data lake.

    Once written, data CANNOT be modified or overwritten.
    """

    def __init__(
        self,
        lake_dir: Union[str, Path] = "data/lake",
        manifest_dir: Union[str, Path] = "data/manifests",
        use_parquet: bool = True,
    ):
        self.lake_dir = Path(lake_dir)
        self.manifest_dir = Path(manifest_dir)
        self.use_parquet = use_parquet and PARQUET_AVAILABLE

        # Create directories
        self.lake_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def freeze_dataframe(
        self,
        df: pd.DataFrame,
        provider: str,
        timeframe: str,
        universe_path: Union[str, Path],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        schema_version: str = SCHEMA_VERSION,
        metadata: Optional[Dict] = None,
        partition_by: Optional[str] = None,  # 'symbol' or 'year' or None
    ) -> DatasetManifest:
        """
        Freeze a DataFrame into the data lake.

        Args:
            df: DataFrame with columns [timestamp, symbol, open, high, low, close, volume]
            provider: Data provider name
            timeframe: Data timeframe (e.g., '1d', '1h')
            universe_path: Path to universe CSV file
            start_date: Override start date (otherwise inferred from data)
            end_date: Override end date (otherwise inferred from data)
            schema_version: Schema version string
            metadata: Optional additional metadata
            partition_by: Partition strategy ('symbol', 'year', or None for single file)

        Returns:
            DatasetManifest for the frozen dataset
        """
        # Validate required columns
        required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Infer date range if not provided
        if start_date is None:
            start_date = df['timestamp'].min().strftime('%Y-%m-%d')
        if end_date is None:
            end_date = df['timestamp'].max().strftime('%Y-%m-%d')

        # Compute dataset ID
        dataset_id = compute_dataset_id(
            provider, timeframe, start_date, end_date,
            universe_path, schema_version
        )

        # Check if already exists
        if dataset_exists(dataset_id, self.manifest_dir):
            existing = find_manifest(dataset_id, self.manifest_dir)
            logger.warning(f"Dataset already exists: {dataset_id}")
            return existing

        # Create dataset directory
        dataset_dir = self.lake_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Write data files
        data_files = []

        if partition_by == 'symbol':
            # One file per symbol
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol].copy()
                filename = f"{symbol}.{'parquet' if self.use_parquet else 'csv'}"
                file_path = dataset_dir / filename
                self._write_file(symbol_df, file_path)
                data_files.append(file_path)

        elif partition_by == 'year':
            # One file per year
            df['_year'] = df['timestamp'].dt.year
            for year in df['_year'].unique():
                year_df = df[df['_year'] == year].drop(columns=['_year']).copy()
                filename = f"{year}.{'parquet' if self.use_parquet else 'csv'}"
                file_path = dataset_dir / filename
                self._write_file(year_df, file_path)
                data_files.append(file_path)
            df = df.drop(columns=['_year'])

        else:
            # Single file
            filename = f"data.{'parquet' if self.use_parquet else 'csv'}"
            file_path = dataset_dir / filename
            self._write_file(df, file_path)
            data_files.append(file_path)

        # Create and save manifest
        manifest = create_manifest(
            provider=provider,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            universe_path=universe_path,
            data_files=data_files,
            schema_version=schema_version,
            metadata=metadata,
        )

        manifest.save(self.manifest_dir)

        logger.info(
            f"Frozen dataset: {dataset_id} "
            f"({manifest.total_rows:,} rows, {manifest.total_symbols} symbols, "
            f"{len(data_files)} files)"
        )

        return manifest

    def _write_file(self, df: pd.DataFrame, path: Path):
        """Write DataFrame to file."""
        if self.use_parquet:
            df.to_parquet(path, index=False, compression='snappy')
        else:
            df.to_csv(path, index=False)


class LakeReader:
    """
    Reads datasets from the frozen data lake.

    Provides integrity verification and efficient querying.
    """

    def __init__(
        self,
        lake_dir: Union[str, Path] = "data/lake",
        manifest_dir: Union[str, Path] = "data/manifests",
    ):
        self.lake_dir = Path(lake_dir)
        self.manifest_dir = Path(manifest_dir)

    def load_dataset(
        self,
        dataset_id: str,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        verify_integrity: bool = False,
    ) -> pd.DataFrame:
        """
        Load a frozen dataset by ID.

        Args:
            dataset_id: The dataset ID to load
            symbols: Optional list of symbols to filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            verify_integrity: Verify file hashes before loading

        Returns:
            DataFrame with OHLCV data
        """
        # Find manifest
        manifest = find_manifest(dataset_id, self.manifest_dir)
        if manifest is None:
            raise FileNotFoundError(f"Dataset not found: {dataset_id}")

        # Verify integrity if requested
        if verify_integrity:
            result = manifest.verify_integrity(self.lake_dir / dataset_id)
            if not result['valid']:
                raise ValueError(
                    f"Dataset integrity check failed: {result['errors']}"
                )

        # Load data files
        dataset_dir = self.lake_dir / dataset_id
        dfs = []

        for file_record in manifest.files:
            file_path = dataset_dir / file_record.path

            if not file_path.exists():
                logger.warning(f"Missing file: {file_path}")
                continue

            # Read file
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)

            dfs.append(df)

        if not dfs:
            raise ValueError(f"No data files found for dataset: {dataset_id}")

        # Combine
        result = pd.concat(dfs, ignore_index=True)

        # Ensure timestamp is datetime
        if 'timestamp' in result.columns:
            result['timestamp'] = pd.to_datetime(result['timestamp'])

        # Apply filters
        if symbols:
            result = result[result['symbol'].isin(symbols)]

        if start_date:
            result = result[result['timestamp'] >= pd.to_datetime(start_date)]

        if end_date:
            result = result[result['timestamp'] <= pd.to_datetime(end_date)]

        # Sort by timestamp and symbol
        result = result.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        return result

    def get_manifest(self, dataset_id: str) -> Optional[DatasetManifest]:
        """Get manifest for a dataset."""
        return find_manifest(dataset_id, self.manifest_dir)

    def list_datasets(self) -> List[DatasetManifest]:
        """List all available datasets."""
        from .manifest import list_manifests
        return list_manifests(self.manifest_dir)

    def verify_dataset(self, dataset_id: str) -> Dict:
        """Verify dataset integrity."""
        manifest = find_manifest(dataset_id, self.manifest_dir)
        if manifest is None:
            return {'valid': False, 'errors': [f"Dataset not found: {dataset_id}"]}

        return manifest.verify_integrity(self.lake_dir / dataset_id)

    def get_coverage_report(self, dataset_id: str) -> Dict:
        """Get coverage report for a dataset."""
        manifest = find_manifest(dataset_id, self.manifest_dir)
        if manifest is None:
            return {'error': f"Dataset not found: {dataset_id}"}

        stats = manifest.get_coverage_stats()

        return {
            'dataset_id': dataset_id,
            'provider': manifest.provider,
            'timeframe': manifest.timeframe,
            'date_range': f"{manifest.start_date} to {manifest.end_date}",
            'years_coverage': stats['years'],
            'symbols': stats['symbols'],
            'total_rows': stats['total_rows'],
            'file_count': stats['file_count'],
            'schema_version': manifest.schema_version,
            'created_at': manifest.created_at,
        }


def quick_load(
    dataset_id: str,
    lake_dir: str = "data/lake",
    manifest_dir: str = "data/manifests",
    **kwargs,
) -> pd.DataFrame:
    """Quick helper to load a dataset."""
    reader = LakeReader(lake_dir, manifest_dir)
    return reader.load_dataset(dataset_id, **kwargs)


def quick_freeze(
    df: pd.DataFrame,
    provider: str,
    timeframe: str,
    universe_path: str,
    lake_dir: str = "data/lake",
    manifest_dir: str = "data/manifests",
    **kwargs,
) -> DatasetManifest:
    """Quick helper to freeze a dataset."""
    writer = LakeWriter(lake_dir, manifest_dir)
    return writer.freeze_dataframe(df, provider, timeframe, universe_path, **kwargs)
