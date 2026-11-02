"""Infer which blocks the current query probably depends on (TF-IDF cosine).

Inferred edges are deliberately weaker than explicit ones: they keep a block reachable,
but only as a *soft* dependency that may still be pruned by attribution when the budget
demands it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from .graph import BlockKind
from .text import cosine, idf, terms, tfidf_vector

if TYPE_CHECKING:  # pragma: no cover
    from .graph import ContextGraph

QUERY_STEP = "__query__"


def infer_edges(
    graph: ContextGraph,
    text: str,
    *,
    step_id: str = QUERY_STEP,
    threshold: float = 0.2,
) -> Dict[str, float]:
    """Link ``step_id`` to every active, unpinned block whose similarity to ``text`` is at
    least ``threshold``. Previous inferred edges of that step are replaced. Returns the
    similarity of every scored block (useful for debugging thresholds).
    """
    candidates = [b for b in graph.blocks() if not b.pinned and b.kind is not BlockKind.SYSTEM]
    query_terms = terms(text)
    step = graph.step(step_id)
    # replace the previous inference, keep any explicit edges the caller added
    for block_id in [bid for bid, e in step.edges.items() if e.kind.value == "inferred"]:
        del step.edges[block_id]
    step.live = True
    if not candidates or not query_terms:
        return {}

    docs = [terms(b.content) for b in candidates]
    weights = idf(docs + [query_terms])
    qv = tfidf_vector(query_terms, weights)
    sims: Dict[str, float] = {}
    for block, doc in zip(candidates, docs, strict=True):
        sims[block.id] = cosine(qv, tfidf_vector(doc, weights))

    # a tool call is linked whenever its result is, and vice versa (same unit)
    by_unit: Dict[str, float] = {}
    for block in candidates:
        by_unit[block.unit] = max(by_unit.get(block.unit, 0.0), sims[block.id])
    for block in candidates:
        score = by_unit[block.unit]
        if score >= threshold:
            graph.link(step_id, [block.id], inferred=True, confidence=min(1.0, score))
    return sims
