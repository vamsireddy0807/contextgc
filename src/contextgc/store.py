"""Cold store: removed blocks live here, intact, until restored."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from .graph import Block


@dataclass
class ColdEntry:
    block: Block
    action: str  # swept | pruned
    reason: str
    score: Optional[float] = None
    policy_version: Optional[str] = None


class ColdStore:
    def __init__(self) -> None:
        self._entries: Dict[str, ColdEntry] = {}

    def put(self, entry: ColdEntry) -> None:
        self._entries[entry.block.id] = entry

    def pop(self, block_id: str) -> ColdEntry:
        return self._entries.pop(block_id)

    def get(self, block_id: str) -> Optional[ColdEntry]:
        return self._entries.get(block_id)

    def entries(self) -> List[ColdEntry]:
        """Entries in original conversation order."""
        return sorted(self._entries.values(), key=lambda e: e.block.seq)

    def ids(self) -> List[str]:
        return [e.block.id for e in self.entries()]

    def __contains__(self, block_id: object) -> bool:
        return block_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[ColdEntry]:
        return iter(self.entries())
