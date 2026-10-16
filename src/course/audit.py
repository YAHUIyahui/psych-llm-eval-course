"""Item analysis and validity audit helpers for evaluation sets."""
from __future__ import annotations

import numpy as np
import pandas as pd


def item_analysis(matrix, item_ids=None):
    """matrix: systems x items of 0/1. Returns pass rate, item-rest
    correlation (discrimination) and flags per item."""
    M = np.asarray(matrix, float)
    total = M.sum(1)
    rows = []
    for j in range(M.shape[1]):
        rest = total - M[:, j]
        r = np.corrcoef(M[:, j], rest)[0, 1] if M[:, j].std() > 0 and rest.std() > 0 else np.nan
        p = M[:, j].mean()
        flag = []
        if p >= 0.95:
            flag.append("too easy")
        if p <= 0.05:
            flag.append("too hard")
        if not np.isnan(r) and r < 0.1:
            flag.append("low discrimination")
        if np.isnan(r):
            flag.append("no variance")
        rows.append({"item": item_ids[j] if item_ids is not None else j,
                     "pass_rate": p, "item_rest_r": r, "flags": ", ".join(flag)})
    return pd.DataFrame(rows)


def cronbach_alpha(matrix):
    """Internal consistency of items (columns) across respondents (rows)."""
    X = np.asarray(matrix, float)
    k = X.shape[1]
    return float(k / (k - 1) * (1 - X.var(0, ddof=1).sum() / X.sum(1).var(ddof=1)))


def saturation_check(scores: dict, ceiling: float = 0.95):
    """scores: system -> (mean, ci_low, ci_high). Saturated when the leaders
    are near the ceiling and their intervals overlap."""
    ranked = sorted(scores.items(), key=lambda kv: -kv[1][0])
    top = ranked[:3]
    near_ceiling = all(m >= ceiling for _, (m, lo, hi) in top)
    overlap = all(top[i][1][1] <= top[0][1][2] for i in range(1, len(top)))
    return {"top_systems": [k for k, _ in top], "near_ceiling": near_ceiling,
            "top_intervals_overlap": overlap, "saturated": near_ceiling and overlap}
