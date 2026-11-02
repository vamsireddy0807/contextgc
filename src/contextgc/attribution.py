"""Causal attribution by ablation: how much does the target depend on each unit of context?

For a scorer ``u(context)`` (higher means the context supports the target better), the
attribution of a unit is ``u(context) - u(context without unit)``. Units are removed whole
(a tool call together with its result), and :func:`contextgc.gc.collect` re-checks the
*marginal* attribution against the current context before each prune, so redundant blocks
are never pruned together on the strength of a stale score.

Three scorers ship with the package:

* :class:`LexicalSupportScorer`: deterministic and offline; the IDF-weighted share of
  target terms present in the context. Good for tests, CI and cheap first-pass pruning.
* :class:`TransformersLogProbScorer`: mean log-probability of the target under a local
  Hugging Face causal LM given the context (``pip install "contextgc[hf]"``).
* :class:`CallableScorer`: wrap any function, such as a call to a hosted model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from .graph import Block
from .text import idf, terms


@runtime_checkable
class Scorer(Protocol):
    def score(self, context: Sequence[Block], target: str, query: Optional[str] = None) -> float:
        """Utility of ``context`` for producing or supporting ``target`` (higher is better)."""
        ...


@dataclass
class UnitAttribution:
    unit: str
    block_ids: Tuple[str, ...]
    tokens: int
    score: Optional[float]
    error: Optional[str] = None


def _finite(x: float) -> float:
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        raise ValueError(f"scorer returned a non-finite value: {x!r}")
    return x


def attribute_units(
    context: Sequence[Block],
    units: Dict[str, List[Block]],
    scorer: Scorer,
    target: str,
    query: Optional[str] = None,
    *,
    max_units: Optional[int] = None,
) -> List[UnitAttribution]:
    """Leave-one-unit-out attribution for ``units`` (a subset of ``context``).

    Never raises because of the scorer: failures are reported per unit in ``error`` with
    ``score=None`` so the caller can keep those units (fail closed).
    """
    prepare = getattr(scorer, "prepare", None)
    ordered = sorted(units.items(), key=lambda kv: -sum(b.tokens for b in kv[1]))
    if max_units is not None:
        selected, skipped = ordered[:max_units], ordered[max_units:]
    else:
        selected, skipped = ordered, []

    def _meta(unit: str, blocks: List[Block]) -> Tuple[Tuple[str, ...], int]:
        return tuple(b.id for b in blocks), sum(b.tokens for b in blocks)

    results: List[UnitAttribution] = []
    try:
        if callable(prepare):
            prepare(context, target, query)
        base = _finite(scorer.score(list(context), target, query))
    except Exception as exc:  # fail closed
        for unit, blocks in ordered:
            ids, tok = _meta(unit, blocks)
            results.append(UnitAttribution(unit, ids, tok, None, f"base score failed: {exc!r}"))
        return results

    for unit, blocks in selected:
        ids, tok = _meta(unit, blocks)
        ablated = [b for b in context if b.unit != unit]
        try:
            value = _finite(scorer.score(ablated, target, query))
            results.append(UnitAttribution(unit, ids, tok, base - value))
        except Exception as exc:  # fail closed
            results.append(UnitAttribution(unit, ids, tok, None, f"ablation failed: {exc!r}"))
    for unit, blocks in skipped:
        ids, tok = _meta(unit, blocks)
        results.append(UnitAttribution(unit, ids, tok, None, "not evaluated: max_attribution_units reached"))
    return results


class LexicalSupportScorer:
    """IDF-weighted fraction of the target's content terms that appear in the context.

    Deterministic and offline. IDF weights are fixed by :meth:`prepare` over the full
    context so scores stay comparable across ablations.
    """

    def __init__(self) -> None:
        self._weights: Dict[str, float] = {}

    def prepare(self, context: Sequence[Block], target: str, query: Optional[str] = None) -> None:
        self._weights = idf([terms(b.content) for b in context] + [terms(target)])

    def score(self, context: Sequence[Block], target: str, query: Optional[str] = None) -> float:
        wanted: Dict[str, int] = {}
        for t in terms(target):
            wanted[t] = wanted.get(t, 0) + 1
        if not wanted:
            return 0.0
        present = set()
        for b in context:
            present.update(terms(b.content))
        default = max(self._weights.values(), default=1.0)
        total = sum(c * self._weights.get(t, default) for t, c in wanted.items())
        got = sum(c * self._weights.get(t, default) for t, c in wanted.items() if t in present)
        return got / total if total else 0.0


class CallableScorer:
    """Wrap ``fn(context_blocks, target, query) -> float`` as a scorer."""

    def __init__(self, fn: Callable[[Sequence[Block], str, Optional[str]], float]) -> None:
        self._fn = fn

    def score(self, context: Sequence[Block], target: str, query: Optional[str] = None) -> float:
        return float(self._fn(context, target, query))


def render_prompt(context: Sequence[Block], query: Optional[str] = None) -> str:
    """Plain-text prompt used by model-based scorers."""
    parts = [f"[{b.kind.value} {b.id}]\n{b.content}" for b in context]
    if query:
        parts.append(f"[question]\n{query}")
    parts.append("[answer]\n")
    return "\n\n".join(parts)


class TransformersLogProbScorer:
    """Mean log-probability of ``target`` given the context under a local causal LM.

    Requires ``pip install "contextgc[hf]"``. You can pass an already loaded ``model`` and
    ``tokenizer`` (any Hugging Face causal LM and its tokenizer) instead of a name.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        model=None,
        tokenizer=None,
        device: Optional[str] = None,
        max_length: int = 4096,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError('TransformersLogProbScorer needs: pip install "contextgc[hf]"') from exc
        self._torch = torch
        if model is None or tokenizer is None:
            if model_name is None:
                raise ValueError("pass model_name, or both model and tokenizer")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    def score(self, context: Sequence[Block], target: str, query: Optional[str] = None) -> float:
        torch = self._torch
        prompt_ids = list(self.tokenizer(render_prompt(context, query))["input_ids"])
        target_ids = list(self.tokenizer(target, add_special_tokens=False)["input_ids"])
        if not target_ids:
            return 0.0
        ids = (prompt_ids + target_ids)[-self.max_length :]  # keep the target, trim the oldest context
        n_target = min(len(target_ids), len(ids) - 1)
        if n_target <= 0:
            return 0.0
        input_ids = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits[0]
        logprobs = torch.log_softmax(logits[:-1].float(), dim=-1)
        token_lp = logprobs.gather(1, input_ids[0, 1:].unsqueeze(1)).squeeze(1)
        return float(token_lp[-n_target:].mean())
