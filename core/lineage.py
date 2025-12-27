"""
Data lineage tracking for KOBE81.

Provides hash-based lineage connecting data, models, decisions, and orders.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os


@dataclass
class LineageRecord:
    """A record in the lineage chain."""
    record_type: str  # dataset, model, decision, order
    record_id: str
    record_hash: str
    parent_hashes: List[str]
    timestamp: str
    metadata: Dict[str, Any]


def compute_file_hash(file_path: Union[str, Path]) -> str:
    """Compute SHA256 hash of a file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_directory_hash(dir_path: Union[str, Path], pattern: str = "*") -> str:
    """
    Compute SHA256 hash of all files in a directory.

    Files are sorted by name for deterministic hashing.
    """
    path = Path(dir_path)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    sha256 = hashlib.sha256()

    # Sort files for deterministic order
    files = sorted(path.glob(pattern))

    for file_path in files:
        if file_path.is_file():
            # Include relative path in hash
            rel_path = file_path.relative_to(path)
            sha256.update(str(rel_path).encode())
            sha256.update(compute_file_hash(file_path).encode())

    return sha256.hexdigest()


def compute_dataset_hash(
    data_paths: List[Union[str, Path]],
    include_content: bool = True,
) -> str:
    """
    Compute hash representing a dataset.

    Args:
        data_paths: List of file or directory paths
        include_content: If True, hash file contents; else hash paths + mtimes
    """
    sha256 = hashlib.sha256()

    for path in sorted(str(p) for p in data_paths):
        p = Path(path)
        if not p.exists():
            # Include missing paths in hash (they affect reproducibility)
            sha256.update(f"MISSING:{path}".encode())
            continue

        if include_content:
            if p.is_file():
                sha256.update(compute_file_hash(p).encode())
            elif p.is_dir():
                sha256.update(compute_directory_hash(p).encode())
        else:
            # Hash path + mtime for faster computation
            mtime = p.stat().st_mtime
            sha256.update(f"{path}:{mtime}".encode())

    return sha256.hexdigest()


def compute_model_hash(
    model_path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Compute hash representing a trained model.

    Includes both model file and configuration.
    """
    sha256 = hashlib.sha256()

    path = Path(model_path)
    if path.exists():
        if path.is_file():
            sha256.update(compute_file_hash(path).encode())
        elif path.is_dir():
            sha256.update(compute_directory_hash(path).encode())
    else:
        sha256.update(f"MISSING:{model_path}".encode())

    if config:
        config_str = json.dumps(config, sort_keys=True, default=str)
        sha256.update(config_str.encode())

    return sha256.hexdigest()


def compute_decision_hash(decision_packet: Dict[str, Any]) -> str:
    """
    Compute hash of a decision packet.

    The packet should be the complete decision context for a trade.
    """
    # Sort keys and serialize deterministically
    packet_str = json.dumps(decision_packet, sort_keys=True, default=str)
    return hashlib.sha256(packet_str.encode()).hexdigest()


def compute_order_hash(
    order_id: str,
    symbol: str,
    side: str,
    qty: float,
    price: float,
    decision_hash: str,
) -> str:
    """Compute hash linking an order to its decision."""
    content = f"{order_id}:{symbol}:{side}:{qty}:{price}:{decision_hash}"
    return hashlib.sha256(content.encode()).hexdigest()


class LineageTracker:
    """
    Tracks data lineage from raw data through orders.

    Maintains a chain of hashes connecting:
    - Dataset → Model → Decision → Order
    """

    def __init__(self, lineage_path: Union[str, Path] = "state/lineage.jsonl"):
        self.lineage_path = Path(lineage_path)
        self._ensure_path()

    def _ensure_path(self) -> None:
        """Ensure lineage file directory exists."""
        self.lineage_path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        record_type: str,
        record_id: str,
        record_hash: str,
        parent_hashes: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageRecord:
        """
        Record a lineage entry.

        Args:
            record_type: Type of record (dataset, model, decision, order)
            record_id: Unique identifier for this record
            record_hash: Hash of the record content
            parent_hashes: Hashes this record depends on
            metadata: Additional metadata
        """
        record = LineageRecord(
            record_type=record_type,
            record_id=record_id,
            record_hash=record_hash,
            parent_hashes=parent_hashes or [],
            timestamp=datetime.utcnow().isoformat() + "Z",
            metadata=metadata or {},
        )

        with open(self.lineage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

        return record

    def record_dataset(
        self,
        dataset_id: str,
        data_paths: List[Union[str, Path]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageRecord:
        """Record a dataset lineage entry."""
        dataset_hash = compute_dataset_hash(data_paths)
        return self.record(
            record_type="dataset",
            record_id=dataset_id,
            record_hash=dataset_hash,
            parent_hashes=[],
            metadata={
                "paths": [str(p) for p in data_paths],
                **(metadata or {}),
            },
        )

    def record_model(
        self,
        model_id: str,
        model_path: Union[str, Path],
        dataset_hash: str,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageRecord:
        """Record a model lineage entry."""
        model_hash = compute_model_hash(model_path, config)
        return self.record(
            record_type="model",
            record_id=model_id,
            record_hash=model_hash,
            parent_hashes=[dataset_hash],
            metadata={
                "path": str(model_path),
                "config": config,
                **(metadata or {}),
            },
        )

    def record_decision(
        self,
        decision_id: str,
        decision_packet: Dict[str, Any],
        model_hash: str,
        dataset_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageRecord:
        """Record a decision lineage entry."""
        decision_hash = compute_decision_hash(decision_packet)
        return self.record(
            record_type="decision",
            record_id=decision_id,
            record_hash=decision_hash,
            parent_hashes=[model_hash, dataset_hash],
            metadata=metadata,
        )

    def record_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        decision_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageRecord:
        """Record an order lineage entry."""
        order_hash = compute_order_hash(
            order_id, symbol, side, qty, price, decision_hash
        )
        return self.record(
            record_type="order",
            record_id=order_id,
            record_hash=order_hash,
            parent_hashes=[decision_hash],
            metadata={
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                **(metadata or {}),
            },
        )

    def get_lineage(self, record_hash: str) -> List[LineageRecord]:
        """Get full lineage chain for a record hash."""
        records: Dict[str, LineageRecord] = {}

        # Load all records
        if self.lineage_path.exists():
            with open(self.lineage_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        records[data["record_hash"]] = LineageRecord(**data)

        # Build lineage chain
        chain = []
        visited = set()
        queue = [record_hash]

        while queue:
            h = queue.pop(0)
            if h in visited or h not in records:
                continue
            visited.add(h)
            record = records[h]
            chain.append(record)
            queue.extend(record.parent_hashes)

        return chain

    def verify_lineage(self, order_hash: str) -> Dict[str, Any]:
        """
        Verify lineage integrity for an order.

        Returns verification results including any missing or broken links.
        """
        chain = self.get_lineage(order_hash)

        if not chain:
            return {
                "valid": False,
                "error": "No lineage found for order",
                "chain": [],
            }

        # Check for complete chain
        has_order = any(r.record_type == "order" for r in chain)
        has_decision = any(r.record_type == "decision" for r in chain)
        has_model = any(r.record_type == "model" for r in chain)
        has_dataset = any(r.record_type == "dataset" for r in chain)

        missing = []
        if not has_order:
            missing.append("order")
        if not has_decision:
            missing.append("decision")
        if not has_model:
            missing.append("model")
        if not has_dataset:
            missing.append("dataset")

        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "chain_length": len(chain),
            "chain": [asdict(r) for r in chain],
        }


def link_lineage(
    order_id: str,
    symbol: str,
    side: str,
    qty: float,
    price: float,
    decision_packet: Dict[str, Any],
    model_path: Optional[Union[str, Path]] = None,
    dataset_paths: Optional[List[Union[str, Path]]] = None,
    tracker: Optional[LineageTracker] = None,
) -> str:
    """
    Convenience function to create complete lineage for an order.

    Returns the order hash.
    """
    if tracker is None:
        tracker = LineageTracker()

    # Compute hashes
    dataset_hash = ""
    if dataset_paths:
        dataset_hash = compute_dataset_hash(dataset_paths)
        tracker.record_dataset(
            dataset_id=f"dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            data_paths=dataset_paths,
        )

    model_hash = ""
    if model_path:
        model_hash = compute_model_hash(model_path)
        tracker.record_model(
            model_id=f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            model_path=model_path,
            dataset_hash=dataset_hash,
        )

    decision_hash = compute_decision_hash(decision_packet)
    decision_id = decision_packet.get("run_id", f"decision_{order_id}")
    tracker.record_decision(
        decision_id=decision_id,
        decision_packet=decision_packet,
        model_hash=model_hash,
        dataset_hash=dataset_hash,
    )

    order_record = tracker.record_order(
        order_id=order_id,
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        decision_hash=decision_hash,
    )

    return order_record.record_hash
