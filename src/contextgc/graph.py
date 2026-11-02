"""The context graph: blocks, dependency edges, steps, cold store and audit log."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Union

from .audit import AuditEntry, AuditLog
from .errors import DuplicateBlock, UnknownBlock
from .store import ColdEntry, ColdStore
from .tokens import TokenCounter, approx_token_count

if TYPE_CHECKING:  # pragma: no cover
    from .attribution import Scorer
    from .gc import CollectReport
    from .policy import Policy


class BlockKind(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RETRIEVED = "retrieved"
    NOTE = "note"


class EdgeKind(str, Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


@dataclass
class Block:
    """One unit of context: a message, a tool call or result, a retrieved chunk or a note."""

    id: str
    kind: BlockKind
    content: str
    tokens: int
    seq: int
    pinned: bool = False
    depends_on: tuple = ()
    group: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def unit(self) -> str:
        """Atomic removal unit. Blocks sharing a group (a tool call and its result) go together."""
        return self.group or self.id

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "content": self.content,
            "tokens": self.tokens,
            "seq": self.seq,
            "pinned": self.pinned,
            "depends_on": list(self.depends_on),
            "group": self.group,
            "meta": self.meta,
        }


@dataclass
class Edge:
    block_id: str
    kind: EdgeKind = EdgeKind.EXPLICIT
    confidence: float = 1.0


@dataclass
class Step:
    """Something that consumes context, such as a model call, an answer or a tool invocation.

    While a step is live, every block it explicitly uses (and everything those blocks
    depend on) is protected from collection.
    """

    id: str
    edges: Dict[str, Edge] = field(default_factory=dict)
    live: bool = True


KindLike = Union[BlockKind, str]


class ContextGraph:
    """Agent context modelled as a dependency graph with garbage collection.

    Example::

        g = ContextGraph(budget_tokens=4000)
        g.add("You are a careful research agent.", kind="system", pinned=True)
        call = g.add_tool_call("search", {"q": "Aurora launch code"})
        result = g.add_tool_result(call, "The Aurora launch code is K7-29.")
        g.link("answer#1", [result])          # explicit dependency: protected
        report = g.collect(Policy(keep_recent=2))
        messages = g.render("openai")
    """

    def __init__(
        self,
        budget_tokens: Optional[int] = None,
        *,
        token_counter: Optional[TokenCounter] = None,
        audit: Optional[AuditLog] = None,
    ) -> None:
        if budget_tokens is not None and budget_tokens < 0:
            raise ValueError("budget_tokens must be >= 0")
        self.budget_tokens = budget_tokens
        self.count_tokens: TokenCounter = token_counter or approx_token_count
        self.cold = ColdStore()
        self.audit = audit if audit is not None else AuditLog()
        self._blocks: Dict[str, Block] = {}
        self._steps: Dict[str, Step] = {}
        self._seq = itertools.count()

    # ------------------------------------------------------------------ adding
    def add(
        self,
        content: str,
        kind: KindLike = BlockKind.NOTE,
        *,
        id: Optional[str] = None,
        pinned: bool = False,
        depends_on: Iterable[str] = (),
        group: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        tokens: Optional[int] = None,
    ) -> str:
        """Add a block and return its id.

        ``depends_on`` lists blocks this one was derived from; if one of them is in the
        cold store it is restored first, so a live block never points at removed context.
        """
        kind = BlockKind(kind)
        seq = next(self._seq)
        block_id = id if id is not None else f"{kind.value}#{seq}"
        if block_id in self._blocks or block_id in self.cold:
            raise DuplicateBlock(block_id)
        deps = tuple(depends_on)
        for dep in deps:
            self._ensure_active(dep, reason=f"dependency of new block {block_id!r}")
        if tokens is None:
            tokens = self.count_tokens(content)
        if tokens < 0:
            raise ValueError("tokens must be >= 0")
        self._blocks[block_id] = Block(
            id=block_id,
            kind=kind,
            content=content,
            tokens=int(tokens),
            seq=seq,
            pinned=pinned,
            depends_on=deps,
            group=group,
            meta=dict(meta or {}),
        )
        return block_id

    def add_message(self, role: KindLike, content: str, **kwargs: Any) -> str:
        """Add a chat message. ``role`` is system, user or assistant."""
        kind = BlockKind(role)
        if kind not in (BlockKind.SYSTEM, BlockKind.USER, BlockKind.ASSISTANT):
            raise ValueError("add_message role must be system, user or assistant")
        if kind is BlockKind.SYSTEM:
            kwargs.setdefault("pinned", True)
        return self.add(content, kind, **kwargs)

    def add_tool_call(self, name: str, arguments: Optional[Dict[str, Any]] = None, *, id: Optional[str] = None) -> str:
        """Add a tool call. Its result (added with ``add_tool_result``) shares its removal unit."""
        arguments = dict(arguments or {})
        content = f"{name}({json.dumps(arguments, sort_keys=True)})"
        block_id = self.add(
            content,
            BlockKind.TOOL_CALL,
            id=id,
            meta={"name": name, "arguments": arguments},
        )
        self._blocks[block_id].group = block_id
        return block_id

    def add_tool_result(self, call_id: str, content: str, *, id: Optional[str] = None) -> str:
        """Add the result of ``call_id``. Call and result are always kept or removed together."""
        self._ensure_active(call_id, reason="tool call of new result")
        call = self._blocks[call_id]
        if call.kind is not BlockKind.TOOL_CALL:
            raise ValueError(f"{call_id!r} is not a tool call")
        return self.add(
            content,
            BlockKind.TOOL_RESULT,
            id=id,
            depends_on=(call_id,),
            group=call.unit,
            meta={"tool_call_id": call_id, "name": call.meta.get("name")},
        )

    # ------------------------------------------------------------------- steps
    def step(self, step_id: str) -> Step:
        if step_id not in self._steps:
            self._steps[step_id] = Step(step_id)
        return self._steps[step_id]

    def link(
        self,
        step: str,
        uses: Iterable[str],
        *,
        inferred: bool = False,
        confidence: float = 1.0,
    ) -> Step:
        """Record that ``step`` uses these blocks.

        Explicit edges protect a block for as long as the step is live. Inferred edges
        only make a block a candidate to keep; an explicit edge is never downgraded.
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        st = self.step(step)
        kind = EdgeKind.INFERRED if inferred else EdgeKind.EXPLICIT
        for block_id in uses:
            self._ensure_active(block_id, reason=f"used by step {step!r}")
            existing = st.edges.get(block_id)
            if existing is not None and existing.kind is EdgeKind.EXPLICIT and inferred:
                continue
            st.edges[block_id] = Edge(block_id, kind, confidence if inferred else 1.0)
        return st

    def complete(self, step: str) -> None:
        """Mark a step finished; its blocks are no longer protected by it."""
        self.step(step).live = False

    def steps(self) -> List[Step]:
        return list(self._steps.values())

    def live_steps(self) -> List[Step]:
        return [s for s in self._steps.values() if s.live]

    # ----------------------------------------------------------------- queries
    def get(self, block_id: str) -> Block:
        try:
            return self._blocks[block_id]
        except KeyError:
            raise UnknownBlock(block_id) from None

    def blocks(self) -> List[Block]:
        """Active blocks in conversation order."""
        return sorted(self._blocks.values(), key=lambda b: b.seq)

    def units(self) -> Dict[str, List[Block]]:
        """Active blocks grouped by removal unit, in order of first appearance."""
        out: Dict[str, List[Block]] = {}
        for b in self.blocks():
            out.setdefault(b.unit, []).append(b)
        return out

    @property
    def tokens(self) -> int:
        return sum(b.tokens for b in self._blocks.values())

    def __contains__(self, block_id: object) -> bool:
        return block_id in self._blocks

    def __len__(self) -> int:
        return len(self._blocks)

    def pin(self, block_id: str) -> None:
        self._ensure_active(block_id, reason="pinned")
        self._blocks[block_id].pinned = True

    def unpin(self, block_id: str) -> None:
        self.get(block_id).pinned = False

    def latest(self, kind: KindLike) -> Optional[Block]:
        kind = BlockKind(kind)
        for b in reversed(self.blocks()):
            if b.kind is kind:
                return b
        return None

    # --------------------------------------------------------- remove/restore
    def _move_to_cold(
        self,
        block_ids: Sequence[str],
        *,
        action: str,
        reason: str,
        score: Optional[float] = None,
        policy_version: Optional[str] = None,
    ) -> List[str]:
        moved: List[str] = []
        for block_id in block_ids:
            block = self._blocks.pop(block_id)
            self.cold.put(ColdEntry(block, action, reason, score, policy_version))
            self.audit.record(
                action=action,
                block_id=block_id,
                kind=block.kind.value,
                tokens=block.tokens,
                reason=reason,
                score=score,
                policy_version=policy_version,
            )
            moved.append(block_id)
        return moved

    def restore(self, block_id: str, *, reason: str = "manual restore") -> List[str]:
        """Bring a removed block (and the rest of its unit) back. Returns restored ids."""
        if block_id in self._blocks:
            return []
        entry = self.cold.get(block_id)
        if entry is None:
            raise UnknownBlock(block_id)
        unit = entry.block.unit
        ids = [e.block.id for e in self.cold.entries() if e.block.unit == unit]
        restored: List[str] = []
        for bid in ids:
            e = self.cold.pop(bid)
            self._blocks[bid] = e.block
            self.audit.record(
                action="restored",
                block_id=bid,
                kind=e.block.kind.value,
                tokens=e.block.tokens,
                reason=reason,
            )
            restored.append(bid)
        # dependencies of a restored block must be active too
        for bid in restored:
            for dep in self._blocks[bid].depends_on:
                self._ensure_active(dep, reason=f"dependency of restored block {bid!r}")
        return restored

    def _ensure_active(self, block_id: str, *, reason: str) -> None:
        if block_id in self._blocks:
            return
        if block_id in self.cold:
            self.restore(block_id, reason=reason)
            return
        raise UnknownBlock(block_id)

    # ------------------------------------------------------------ high level
    def collect(
        self,
        policy: Optional[Policy] = None,
        *,
        scorer: Optional[Scorer] = None,
        target: Optional[str] = None,
        query: Optional[str] = None,
    ) -> CollectReport:
        """Run garbage collection. See :func:`contextgc.gc.collect`."""
        from .gc import collect

        return collect(self, policy, scorer=scorer, target=target, query=query)

    def render(self, style: str = "openai", *, tombstones: bool = True) -> Any:
        """Render active context. ``style`` is ``openai``, ``anthropic`` or ``text``."""
        from .render import render

        return render(self, style=style, tombstones=tombstones)

    def audit_log(self) -> List[AuditEntry]:
        return self.audit.entries()

    # --------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        return {
            "budget_tokens": self.budget_tokens,
            "blocks": [b.to_dict() for b in self.blocks()],
            "cold": [
                {
                    "block": e.block.to_dict(),
                    "action": e.action,
                    "reason": e.reason,
                    "score": e.score,
                    "policy_version": e.policy_version,
                }
                for e in self.cold.entries()
            ],
            "steps": [
                {
                    "id": s.id,
                    "live": s.live,
                    "edges": [
                        {"block_id": e.block_id, "kind": e.kind.value, "confidence": e.confidence}
                        for e in s.edges.values()
                    ],
                }
                for s in self._steps.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict, *, token_counter: Optional[TokenCounter] = None) -> ContextGraph:
        g = cls(data.get("budget_tokens"), token_counter=token_counter)
        max_seq = -1

        def _block(d: dict) -> Block:
            return Block(
                id=d["id"],
                kind=BlockKind(d["kind"]),
                content=d["content"],
                tokens=int(d["tokens"]),
                seq=int(d["seq"]),
                pinned=bool(d.get("pinned", False)),
                depends_on=tuple(d.get("depends_on", ())),
                group=d.get("group"),
                meta=dict(d.get("meta") or {}),
            )

        for d in data.get("blocks", []):
            b = _block(d)
            g._blocks[b.id] = b
            max_seq = max(max_seq, b.seq)
        for c in data.get("cold", []):
            b = _block(c["block"])
            g.cold.put(ColdEntry(b, c["action"], c["reason"], c.get("score"), c.get("policy_version")))
            max_seq = max(max_seq, b.seq)
        for s in data.get("steps", []):
            st = Step(s["id"], live=bool(s.get("live", True)))
            for e in s.get("edges", []):
                st.edges[e["block_id"]] = Edge(
                    e["block_id"], EdgeKind(e.get("kind", "explicit")), float(e.get("confidence", 1.0))
                )
            g._steps[st.id] = st
        g._seq = itertools.count(max_seq + 1)
        return g

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs: Any) -> ContextGraph:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), **kwargs)
