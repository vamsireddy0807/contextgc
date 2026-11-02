"""Exceptions raised by ContextGC."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .gc import CollectReport


class ContextGCError(Exception):
    """Base class for all ContextGC errors."""


class UnknownBlock(ContextGCError, KeyError):
    """A block id was referenced that is neither active nor in the cold store."""

    def __init__(self, block_id: str) -> None:
        self.block_id = block_id
        super().__init__(f"unknown block id: {block_id!r}")

    def __str__(self) -> str:  # KeyError would otherwise quote the message
        return f"unknown block id: {self.block_id!r}"


class DuplicateBlock(ContextGCError, ValueError):
    """A block id was added twice."""

    def __init__(self, block_id: str) -> None:
        self.block_id = block_id
        super().__init__(f"block id already exists: {block_id!r}")


class BudgetExceeded(ContextGCError):
    """The token budget could not be met without breaking a safety rule.

    ContextGC never silently truncates protected context. When the budget cannot be
    met, ``collect()`` raises this error (unless the policy says ``on_over_budget="warn"``).
    Everything already swept or pruned before the error is still restorable, and the
    partial report is attached as ``.report``.
    """

    def __init__(self, report: CollectReport) -> None:
        self.report = report
        super().__init__(
            f"budget of {report.budget_tokens} tokens not met: {report.tokens_after} tokens "
            f"remain after collection ({len(report.swept)} swept, {len(report.pruned)} pruned)"
        )
