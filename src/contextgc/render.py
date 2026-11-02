"""Render the active context for a model API.

* ``openai``: a list of chat messages; tool calls become ``assistant.tool_calls`` and
  results become ``role: tool`` messages. Consecutive tool calls are merged into one
  assistant message, as the API requires.
* ``anthropic``: ``{"system": str, "messages": [...]}`` with ``tool_use`` and
  ``tool_result`` content blocks; consecutive same-role messages are merged and
  ``tool_result`` blocks are placed first in a user turn.
* ``text``: a readable transcript with tombstones inline, for logs and debugging.

In the API styles, tombstones for removed context are summarised in one system note
(never inline), so they can never split a tool call from its result.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List

from .graph import Block, BlockKind

if TYPE_CHECKING:  # pragma: no cover
    from .graph import ContextGraph
    from .store import ColdEntry


def _tombstone_lines(entries: List[ColdEntry]) -> List[str]:
    lines: List[str] = []
    seen_units = set()
    for e in entries:
        unit = e.block.unit
        if unit in seen_units:
            continue
        seen_units.add(unit)
        members = [x for x in entries if x.block.unit == unit]
        tokens = sum(x.block.tokens for x in members)
        kinds = "+".join(sorted({x.block.kind.value for x in members}))
        ids = ", ".join(x.block.id for x in members)
        lines.append(f"- {kinds} [{ids}] ({tokens} tokens) {e.action}: {e.reason}")
    return lines


def _tombstone_note(graph: ContextGraph) -> str:
    entries = graph.cold.entries()
    if not entries:
        return ""
    return (
        "[contextgc] Some earlier context was removed to fit the budget and can be restored "
        "on request:\n" + "\n".join(_tombstone_lines(entries))
    )


def _labelled(b: Block) -> str:
    return f"[{b.kind.value} {b.id}]\n{b.content}"


def render(graph: ContextGraph, style: str = "openai", *, tombstones: bool = True) -> Any:
    if style == "openai":
        return _render_openai(graph, tombstones)
    if style == "anthropic":
        return _render_anthropic(graph, tombstones)
    if style == "text":
        return _render_text(graph, tombstones)
    raise ValueError("style must be 'openai', 'anthropic' or 'text'")


def _render_openai(graph: ContextGraph, tombstones: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    note = _tombstone_note(graph) if tombstones else ""
    blocks = graph.blocks()
    systems = [b for b in blocks if b.kind is BlockKind.SYSTEM]
    for b in systems:
        out.append({"role": "system", "content": b.content})
    if note:
        out.append({"role": "system", "content": note})
    for b in blocks:
        if b.kind is BlockKind.SYSTEM:
            continue
        if b.kind is BlockKind.TOOL_CALL:
            call = {
                "id": b.id,
                "type": "function",
                "function": {
                    "name": b.meta.get("name", "tool"),
                    "arguments": json.dumps(b.meta.get("arguments", {}), sort_keys=True),
                },
            }
            prev = out[-1] if out else None
            if prev and prev.get("role") == "assistant" and prev.get("tool_calls") and prev.get("content") is None:
                prev["tool_calls"].append(call)
            else:
                out.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif b.kind is BlockKind.TOOL_RESULT:
            out.append({"role": "tool", "tool_call_id": b.meta.get("tool_call_id", b.unit), "content": b.content})
        elif b.kind in (BlockKind.USER, BlockKind.ASSISTANT):
            out.append({"role": b.kind.value, "content": b.content})
        else:  # retrieved chunks and notes are given to the model as user-side context
            out.append({"role": "user", "content": _labelled(b)})
    return out


def _render_anthropic(graph: ContextGraph, tombstones: bool) -> Dict[str, Any]:
    blocks = graph.blocks()
    system_parts = [b.content for b in blocks if b.kind is BlockKind.SYSTEM]
    if tombstones:
        note = _tombstone_note(graph)
        if note:
            system_parts.append(note)
    messages: List[Dict[str, Any]] = []

    def push(role: str, block: Dict[str, Any]) -> None:
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].append(block)
        else:
            messages.append({"role": role, "content": [block]})

    for b in blocks:
        if b.kind is BlockKind.SYSTEM:
            continue
        if b.kind is BlockKind.TOOL_CALL:
            push(
                "assistant",
                {
                    "type": "tool_use",
                    "id": b.id,
                    "name": b.meta.get("name", "tool"),
                    "input": b.meta.get("arguments", {}),
                },
            )
        elif b.kind is BlockKind.TOOL_RESULT:
            push(
                "user",
                {"type": "tool_result", "tool_use_id": b.meta.get("tool_call_id", b.unit), "content": b.content},
            )
        elif b.kind is BlockKind.ASSISTANT:
            push("assistant", {"type": "text", "text": b.content})
        elif b.kind is BlockKind.USER:
            push("user", {"type": "text", "text": b.content})
        else:
            push("user", {"type": "text", "text": _labelled(b)})

    for m in messages:  # tool_result blocks must lead a user turn
        if m["role"] == "user":
            m["content"].sort(key=lambda c: 0 if c.get("type") == "tool_result" else 1)
    return {"system": "\n\n".join(system_parts), "messages": messages}


def _render_text(graph: ContextGraph, tombstones: bool) -> str:
    items = [(b.seq, "active", b) for b in graph.blocks()]
    if tombstones:
        items += [(e.block.seq, "cold", e) for e in graph.cold.entries()]
    items.sort(key=lambda x: x[0])
    parts: List[str] = []
    emitted_units = set()
    cold_entries = graph.cold.entries()
    for _, state, obj in items:
        if state == "active":
            parts.append(_labelled(obj))
            continue
        unit = obj.block.unit
        if unit in emitted_units:
            continue
        emitted_units.add(unit)
        members = [e for e in cold_entries if e.block.unit == unit]
        parts.append("[contextgc tombstone]\n" + "\n".join(_tombstone_lines(members)))
    return "\n\n".join(parts)
