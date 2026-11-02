"""ContextGC: dependency-aware context runtime for LLM agents.

Semantic garbage collection, dependency tracking and causal attribution for context
pruning, with explicit failure boundaries: protected context is never silently dropped,
every removal is audited, and every removal is reversible.
"""

from .attribution import (
    CallableScorer,
    LexicalSupportScorer,
    Scorer,
    TransformersLogProbScorer,
    UnitAttribution,
    attribute_units,
)
from .audit import AuditEntry, AuditLog
from .errors import BudgetExceeded, ContextGCError, DuplicateBlock, UnknownBlock
from .gc import CollectReport, collect, reachability
from .graph import Block, BlockKind, ContextGraph, Edge, EdgeKind, Step
from .infer import infer_edges
from .policy import Policy
from .render import render
from .tokens import approx_token_count

__version__ = "0.1.0"

__all__ = [
    "AuditEntry",
    "AuditLog",
    "Block",
    "BlockKind",
    "BudgetExceeded",
    "CallableScorer",
    "CollectReport",
    "ContextGCError",
    "ContextGraph",
    "DuplicateBlock",
    "Edge",
    "EdgeKind",
    "LexicalSupportScorer",
    "Policy",
    "Scorer",
    "Step",
    "TransformersLogProbScorer",
    "UnitAttribution",
    "UnknownBlock",
    "approx_token_count",
    "attribute_units",
    "collect",
    "infer_edges",
    "reachability",
    "render",
    "__version__",
]
