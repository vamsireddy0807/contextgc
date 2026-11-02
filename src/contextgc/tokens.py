"""Token counting.

ContextGC only needs consistent, approximate counts to enforce a budget. The default
counter is a dependency-free heuristic (about 4 characters per token, the usual rule of
thumb for English with BPE tokenizers). Pass your model's real tokenizer for exact
budgets, for example::

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    graph = ContextGraph(budget_tokens=8000, token_counter=lambda s: len(enc.encode(s)))
"""

from __future__ import annotations

import math
from typing import Callable

TokenCounter = Callable[[str], int]


def approx_token_count(text: str) -> int:
    """Approximate token count: ceil(characters / 4), and at least 1 for non-empty text."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))
