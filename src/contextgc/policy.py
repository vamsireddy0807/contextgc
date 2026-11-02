"""Versioned collection policies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal, Optional

SweepMode = Literal["lazy", "eager"]
AttributionMode = Literal["none", "ablation"]
OverBudget = Literal["raise", "warn"]


@dataclass(frozen=True)
class Policy:
    """How ``collect()`` decides what to keep.

    Attributes:
        budget_tokens: Token budget for this collection. ``None`` falls back to the
            graph's own budget; if both are ``None`` there is no budget and only eager
            sweeping removes anything.
        keep_recent: The newest N blocks are always treated as roots.
        sweep: ``"lazy"`` sweeps unreachable blocks oldest-first only while over budget;
            ``"eager"`` sweeps every unreachable block on every collection.
        attribution: ``"ablation"`` scores inferred-only blocks by leave-one-out ablation
            with a scorer before pruning them; ``"none"`` prunes them by lowest inferred
            confidence, oldest first.
        min_attribution: Blocks whose attribution is at or above this value are never
            pruned in ablation mode.
        max_attribution_units: Cap on how many units are scored per collection (each
            costs one scorer call). Units beyond the cap are kept, not pruned.
        infer_edges: Infer dependencies from the current query (TF-IDF cosine) before
            marking.
        infer_threshold: Minimum cosine similarity for an inferred edge.
        tombstones: Render a short note where context was removed.
        on_over_budget: ``"raise"`` raises ``BudgetExceeded``; ``"warn"`` returns a report
            with ``budget_met=False``.
        name: Free-form label recorded in the audit log.
    """

    budget_tokens: Optional[int] = None
    keep_recent: int = 4
    sweep: SweepMode = "lazy"
    attribution: AttributionMode = "none"
    min_attribution: float = 0.05
    max_attribution_units: int = 32
    infer_edges: bool = False
    infer_threshold: float = 0.2
    tombstones: bool = True
    on_over_budget: OverBudget = "raise"
    name: str = "default"

    def __post_init__(self) -> None:
        if self.budget_tokens is not None and self.budget_tokens < 0:
            raise ValueError("budget_tokens must be >= 0")
        if self.keep_recent < 0:
            raise ValueError("keep_recent must be >= 0")
        if self.sweep not in ("lazy", "eager"):
            raise ValueError("sweep must be 'lazy' or 'eager'")
        if self.attribution not in ("none", "ablation"):
            raise ValueError("attribution must be 'none' or 'ablation'")
        if self.max_attribution_units < 0:
            raise ValueError("max_attribution_units must be >= 0")
        if not 0.0 <= self.infer_threshold <= 1.0:
            raise ValueError("infer_threshold must be within [0, 1]")
        if self.on_over_budget not in ("raise", "warn"):
            raise ValueError("on_over_budget must be 'raise' or 'warn'")

    @property
    def version_id(self) -> str:
        """Stable 12-character hash of every field; stamped on each audit entry."""
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
