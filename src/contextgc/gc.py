"""Mark, sweep and prune.

Collection runs in three phases:

1. **Mark.** Roots are pinned blocks, the newest ``keep_recent`` blocks, and every block a
   live step uses through an *explicit* edge. Their transitive closure over ``depends_on``
   and removal units is *hard*-reachable and is never removed. Blocks reached only through
   *inferred* edges are *soft*-reachable.
2. **Sweep.** Unreachable units move to the cold store, oldest first (``lazy``: only while
   over budget; ``eager``: always).
3. **Prune.** If still over budget, soft-reachable units are pruned: by ablation
   attribution (lowest first, and only below ``min_attribution``, re-checked against the
   current context before each prune) or, without a scorer, by lowest inferred confidence.

If the budget still is not met, ``BudgetExceeded`` is raised: protected context is never
silently truncated. Every removal is audited and reversible with ``graph.restore``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from .attribution import Scorer, attribute_units
from .errors import BudgetExceeded
from .graph import Block, BlockKind, EdgeKind
from .infer import infer_edges
from .policy import Policy

if TYPE_CHECKING:  # pragma: no cover
    from .graph import ContextGraph


@dataclass
class CollectReport:
    policy_version: str
    budget_tokens: Optional[int]
    tokens_before: int
    tokens_after: int
    protected: List[str] = field(default_factory=list)
    soft: List[str] = field(default_factory=list)
    swept: List[str] = field(default_factory=list)
    pruned: List[str] = field(default_factory=list)
    kept_on_error: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    budget_met: bool = True

    @property
    def tokens_removed(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def reduction(self) -> float:
        return self.tokens_removed / self.tokens_before if self.tokens_before else 0.0

    def to_dict(self) -> dict:
        return {
            "policy_version": self.policy_version,
            "budget_tokens": self.budget_tokens,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_removed": self.tokens_removed,
            "reduction": round(self.reduction, 4),
            "protected": self.protected,
            "soft": self.soft,
            "swept": self.swept,
            "pruned": self.pruned,
            "kept_on_error": self.kept_on_error,
            "scores": self.scores,
            "budget_met": self.budget_met,
        }

    def summary(self) -> str:
        budget = "none" if self.budget_tokens is None else str(self.budget_tokens)
        return (
            f"tokens {self.tokens_before} -> {self.tokens_after} "
            f"({self.reduction:.1%} removed, budget {budget}, met={self.budget_met}); "
            f"protected {len(self.protected)}, soft {len(self.soft)}, swept {len(self.swept)}, "
            f"pruned {len(self.pruned)}, kept on error {len(self.kept_on_error)} "
            f"[policy {self.policy_version}]"
        )


def _closure(seeds: Set[str], graph: ContextGraph) -> Set[str]:
    active = {b.id: b for b in graph.blocks()}
    units = graph.units()
    seen: Set[str] = set()
    stack = [s for s in seeds if s in active]
    while stack:
        block_id = stack.pop()
        if block_id in seen or block_id not in active:
            continue
        seen.add(block_id)
        block = active[block_id]
        stack.extend(block.depends_on)
        stack.extend(b.id for b in units.get(block.unit, ()))
    return seen


def reachability(graph: ContextGraph, policy: Policy) -> Tuple[Set[str], Set[str]]:
    """Return ``(hard, soft)`` reachable block ids."""
    ordered = graph.blocks()
    active = {b.id for b in ordered}
    roots: Set[str] = {b.id for b in ordered if b.pinned}
    if policy.keep_recent:
        roots.update(b.id for b in ordered[-policy.keep_recent :])
    soft_seeds: Set[str] = set()
    for step in graph.live_steps():
        for edge in step.edges.values():
            if edge.block_id not in active:
                continue
            if edge.kind is EdgeKind.EXPLICIT:
                roots.add(edge.block_id)
            else:
                soft_seeds.add(edge.block_id)
    hard = _closure(roots, graph)
    soft = _closure(soft_seeds - hard, graph) - hard
    return hard, soft


def _default_target(graph: ContextGraph) -> Optional[str]:
    latest = graph.latest(BlockKind.USER)
    return latest.content if latest else None


def _inferred_confidence(graph: ContextGraph, blocks: List[Block]) -> float:
    ids = {b.id for b in blocks}
    best = 0.0
    for step in graph.live_steps():
        for edge in step.edges.values():
            if edge.block_id in ids and edge.kind is EdgeKind.INFERRED:
                best = max(best, edge.confidence)
    return best


def collect(
    graph: ContextGraph,
    policy: Optional[Policy] = None,
    *,
    scorer: Optional[Scorer] = None,
    target: Optional[str] = None,
    query: Optional[str] = None,
) -> CollectReport:
    """Garbage-collect ``graph`` under ``policy``. See the module docstring for the phases."""
    policy = policy or Policy()
    version = policy.version_id
    budget = policy.budget_tokens if policy.budget_tokens is not None else graph.budget_tokens
    report = CollectReport(version, budget, graph.tokens, graph.tokens)

    def over() -> bool:
        return budget is not None and graph.tokens > budget

    if policy.attribution == "ablation" and scorer is None:
        raise ValueError("attribution='ablation' requires a scorer")

    probe = query or target or _default_target(graph)
    if policy.infer_edges and probe:
        infer_edges(graph, probe, threshold=policy.infer_threshold)

    hard, soft = reachability(graph, policy)
    report.protected = [b.id for b in graph.blocks() if b.id in hard]
    report.soft = [b.id for b in graph.blocks() if b.id in soft]

    # ---- phase 2: sweep unreachable units, oldest first
    for _unit, blocks in graph.units().items():
        ids = [b.id for b in blocks]
        if any(i in hard or i in soft for i in ids) or any(b.pinned for b in blocks):
            continue
        if policy.sweep == "lazy" and not over():
            break
        report.swept += graph._move_to_cold(ids, action="swept", reason="unreachable", policy_version=version)

    # ---- phase 3: prune soft-reachable units
    if over():
        candidates: Dict[str, List[Block]] = {}
        for unit, blocks in graph.units().items():
            ids = {b.id for b in blocks}
            if ids & hard or any(b.pinned for b in blocks) or not ids & soft:
                continue
            candidates[unit] = blocks

        if candidates and policy.attribution == "ablation":
            assert scorer is not None
            tgt = target or query or _default_target(graph)
            if not tgt:
                raise ValueError("ablation needs a target or query (or a user message in the graph)")
            context = graph.blocks()
            results = attribute_units(context, candidates, scorer, tgt, query, max_units=policy.max_attribution_units)
            for r in results:
                if r.score is not None:
                    for bid in r.block_ids:
                        report.scores[bid] = r.score
                else:
                    report.kept_on_error += list(r.block_ids)
                    for bid in r.block_ids:
                        blk = graph.get(bid)
                        graph.audit.record(
                            action="kept",
                            block_id=bid,
                            kind=blk.kind.value,
                            tokens=blk.tokens,
                            reason=r.error or "not scored",
                            policy_version=version,
                        )
            ranked = sorted(
                (r for r in results if r.score is not None),
                key=lambda r: (r.score, -r.tokens),
            )
            for r in ranked:
                if not over():
                    break
                if r.score >= policy.min_attribution:
                    break
                # marginal re-check against the *current* context (protects redundant copies)
                current = graph.blocks()
                try:
                    prep = getattr(scorer, "prepare", None)
                    if callable(prep):
                        prep(current, tgt, query)
                    now = scorer.score(current, tgt, query)
                    without = scorer.score([b for b in current if b.unit != r.unit], tgt, query)
                    marginal = float(now) - float(without)
                except Exception as exc:  # fail closed
                    report.kept_on_error += list(r.block_ids)
                    for bid in r.block_ids:
                        blk = graph.get(bid)
                        graph.audit.record(
                            action="kept",
                            block_id=bid,
                            kind=blk.kind.value,
                            tokens=blk.tokens,
                            reason=f"marginal check failed: {exc!r}",
                            policy_version=version,
                        )
                    continue
                if marginal >= policy.min_attribution:
                    for bid in r.block_ids:
                        report.scores[bid] = marginal
                    continue
                report.pruned += graph._move_to_cold(
                    list(r.block_ids),
                    action="pruned",
                    reason=f"attribution {marginal:.4f} < {policy.min_attribution}",
                    score=marginal,
                    policy_version=version,
                )
        elif candidates:
            ordered = sorted(
                candidates.items(),
                key=lambda kv: (_inferred_confidence(graph, kv[1]), kv[1][0].seq),
            )
            for _unit, blocks in ordered:
                if not over():
                    break
                conf = _inferred_confidence(graph, blocks)
                report.pruned += graph._move_to_cold(
                    [b.id for b in blocks],
                    action="pruned",
                    reason=f"inferred-only dependency, confidence {conf:.3f}",
                    score=conf,
                    policy_version=version,
                )

    report.tokens_after = graph.tokens
    report.budget_met = not over()
    if not report.budget_met and policy.on_over_budget == "raise":
        raise BudgetExceeded(report)
    return report
