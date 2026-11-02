"""Append-only audit log: every sweep, prune, restore and error-driven keep is recorded."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Union

PathLike = Union[str, Path]


@dataclass(frozen=True)
class AuditEntry:
    seq: int
    timestamp: float
    action: str  # swept | pruned | restored | kept
    block_id: str
    kind: str
    tokens: int
    reason: str
    score: Optional[float] = None
    policy_version: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class AuditLog:
    """In-memory append-only log, optionally mirrored to a JSONL file as entries arrive."""

    def __init__(self, path: Optional[PathLike] = None) -> None:
        self._entries: List[AuditEntry] = []
        self._path = Path(path) if path is not None else None

    def record(
        self,
        *,
        action: str,
        block_id: str,
        kind: str,
        tokens: int,
        reason: str,
        score: Optional[float] = None,
        policy_version: Optional[str] = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            seq=len(self._entries),
            timestamp=time.time(),
            action=action,
            block_id=block_id,
            kind=kind,
            tokens=tokens,
            reason=reason,
            score=score,
            policy_version=policy_version,
        )
        self._entries.append(entry)
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
        return entry

    def entries(self) -> List[AuditEntry]:
        return list(self._entries)

    def __iter__(self) -> Iterator[AuditEntry]:
        return iter(list(self._entries))

    def __len__(self) -> int:
        return len(self._entries)

    def to_jsonl(self, path: PathLike) -> None:
        with Path(path).open("w", encoding="utf-8") as fh:
            for entry in self._entries:
                fh.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")

    @staticmethod
    def read_jsonl(path: PathLike) -> List[AuditEntry]:
        out: List[AuditEntry] = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(AuditEntry(**json.loads(line)))
        return out
