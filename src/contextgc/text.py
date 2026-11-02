"""Small, dependency-free text utilities shared by edge inference and the lexical scorer."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence

_WORD = re.compile(r"[A-Za-z0-9]+(?:[-_'][A-Za-z0-9]+)*")

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into is it
    its itself just me more most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself yourselves
    please tell give show find get also may might must shall""".split()
)


def terms(text: str) -> List[str]:
    """Lower-cased content terms: words of 2+ characters that are not stopwords."""
    out = []
    for w in _WORD.findall(text):
        lw = w.lower()
        if len(lw) >= 2 and lw not in STOPWORDS:
            out.append(lw)
    return out


def idf(documents: Sequence[Iterable[str]]) -> Dict[str, float]:
    """Smoothed inverse document frequency over a set of term lists."""
    n = len(documents)
    df: Counter = Counter()
    for doc in documents:
        df.update(set(doc))
    return {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}


def tfidf_vector(doc_terms: Iterable[str], weights: Dict[str, float]) -> Dict[str, float]:
    tf = Counter(doc_terms)
    vec = {t: c * weights.get(t, 0.0) for t, c in tf.items()}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm == 0:
        return {}
    return {t: v / norm for t, v in vec.items()}


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())
