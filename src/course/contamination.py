"""Simple contamination checks for evaluation items."""
from __future__ import annotations

import difflib
import re


def ngrams(text: str, n: int = 8):
    toks = re.findall(r"\w+", text.lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def ngram_overlap(item: str, corpus, n: int = 8) -> float:
    """Share of the item's n-grams that appear anywhere in the corpus.
    Needs access to the training corpus, which evaluators often lack."""
    g = ngrams(item, n)
    if not g:
        return 0.0
    seen = set().union(*(ngrams(doc, n) for doc in corpus))
    return len(g & seen) / len(g)


def completion_similarity(client, text: str, prefix_frac: float = 0.5) -> float:
    """Give the model the first part of an item and compare its continuation
    with the true second part. High similarity on many items is evidence of
    memorisation, not proof."""
    cut = int(len(text) * prefix_frac)
    prefix, rest = text[:cut], text[cut:]
    out = client.chat([{"role": "user", "content": "Continue this text exactly:\n" + prefix}])
    cont = (out.get("content") or "")[:len(rest)]
    return difflib.SequenceMatcher(None, cont.lower(), rest.lower()).ratio()
