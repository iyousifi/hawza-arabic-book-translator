"""Translation checkpoint and resumption management.

Enables translation runs to survive network drops, process kills, and rate limits
by persisting completed chunk drafts, verifications, and learned terminology.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SavedChunk:
    """A single completed translation chunk saved in a checkpoint."""

    index: int
    chunk_output: str
    final_text: str = ""
    audit_entry: Optional[Dict[str, Any]] = None


@dataclass
class CheckpointData:
    """Full checkpoint state for an active translation run."""

    version: int = 1
    source_hash: str = ""
    total_chunks: int = 0
    target_lang: str = ""
    domain: str = ""
    inline_arabic: bool = False
    verify: bool = False
    chunks: List[SavedChunk] = field(default_factory=list)
    session_terms: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "source_hash": self.source_hash,
            "total_chunks": self.total_chunks,
            "target_lang": self.target_lang,
            "domain": self.domain,
            "inline_arabic": self.inline_arabic,
            "verify": self.verify,
            "chunks": [asdict(c) for c in self.chunks],
            "session_terms": self.session_terms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointData":
        raw_chunks = data.get("chunks", [])
        chunks = [
            SavedChunk(
                index=c["index"],
                chunk_output=c["chunk_output"],
                final_text=c.get("final_text", ""),
                audit_entry=c.get("audit_entry"),
            )
            for c in raw_chunks
        ]
        return cls(
            version=data.get("version", 1),
            source_hash=data.get("source_hash", ""),
            total_chunks=data.get("total_chunks", 0),
            target_lang=data.get("target_lang", ""),
            domain=data.get("domain", ""),
            inline_arabic=data.get("inline_arabic", False),
            verify=data.get("verify", False),
            chunks=chunks,
            session_terms=data.get("session_terms", {}),
        )


def compute_source_hash(chunk_texts: List[str]) -> str:
    """Compute deterministic SHA-256 hash across all source chunk texts."""
    hasher = hashlib.sha256()
    for text in chunk_texts:
        hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def load_checkpoint(checkpoint_path: Path | str) -> Optional[CheckpointData]:
    """Load and parse checkpoint file if present and valid."""
    path = Path(checkpoint_path)
    if not path.is_file():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "chunks" not in data:
            return None
        return CheckpointData.from_dict(data)
    except Exception:
        return None


def is_checkpoint_compatible(
    checkpoint: CheckpointData,
    source_hash: str,
    total_chunks: int,
    target_lang: str,
    domain: str,
    inline_arabic: bool,
    verify: bool,
) -> bool:
    """Validate that the checkpoint matches the current translation configuration."""
    if checkpoint.total_chunks != total_chunks:
        return False
    if checkpoint.source_hash != source_hash:
        return False
    if checkpoint.target_lang.lower() != target_lang.lower():
        return False
    if checkpoint.domain.lower() != domain.lower():
        return False
    if checkpoint.inline_arabic != inline_arabic:
        return False
    if checkpoint.verify != verify:
        return False
    return True


def save_checkpoint(checkpoint_path: Path | str, checkpoint: CheckpointData) -> None:
    """Atomically save checkpoint to disk to guard against crash mid-write."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    data = checkpoint.to_dict()
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    os.replace(temp_path, path)


def remove_checkpoint(checkpoint_path: Path | str) -> None:
    """Safely delete checkpoint file once translation completes."""
    path = Path(checkpoint_path)
    try:
        if path.is_file():
            path.unlink(missing_ok=True)
        # Also clean up lingering temp file if any
        temp_path = path.with_suffix(path.suffix + ".tmp")
        if temp_path.is_file():
            temp_path.unlink(missing_ok=True)
    except Exception:
        pass
