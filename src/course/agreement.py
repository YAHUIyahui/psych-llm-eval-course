"""Inter-rater agreement for two raters (and Krippendorff's alpha for many).

All functions take plain array-likes. Ratings are labels (nominal) or ordered
numbers (ordinal / interval). Missing values in alpha are np.nan.
"""
from __future__ import annotations

import numpy as np


def percent_agreement(a, b) -> float:
    """Raw proportion of identical ratings. Not corrected for chance."""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean(a == b))


def _confusion(a, b, categories):
    idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    return m


def cohen_kappa(a, b, weights: str | None = None, categories=None) -> float:
    """Cohen's kappa. weights: None, 'linear' or 'quadratic' (ordinal data).

    Pass `categories` (ordered) when a category may be absent from a sample,
    for example inside a bootstrap resample.
    """
    a, b = np.asarray(a), np.asarray(b)
    if categories is None:
        categories = sorted(set(a.tolist()) | set(b.tolist()))
    k = len(categories)
    obs = _confusion(a, b, categories) / len(a)
    exp = np.outer(obs.sum(1), obs.sum(0))
    i, j = np.indices((k, k))
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / max(k - 1, 1)
    elif weights == "quadratic":
        w = ((i - j) / max(k - 1, 1)) ** 2
    else:
        raise ValueError("weights must be None, 'linear' or 'quadratic'")
    denom = (w * exp).sum()
    if denom == 0:
        return float("nan")
    return float(1 - (w * obs).sum() / denom)


def gwet_ac1(a, b, categories=None) -> float:
    """Gwet's AC1 (nominal). Less affected than kappa by skewed prevalence."""
    a, b = np.asarray(a), np.asarray(b)
    if categories is None:
        categories = sorted(set(a.tolist()) | set(b.tolist()))
    k = len(categories)
    obs = _confusion(a, b, categories) / len(a)
    pa = float(np.trace(obs))
    pi = (obs.sum(0) + obs.sum(1)) / 2
    pe = float((pi * (1 - pi)).sum() / (k - 1)) if k > 1 else 1.0
    if pe == 1:
        return float("nan")
    return (pa - pe) / (1 - pe)


def krippendorff_alpha(ratings, level: str = "ordinal") -> float:
    """Krippendorff's alpha. ratings: raters x items array, np.nan = missing.

    level: 'nominal', 'ordinal' or 'interval'. Follows Krippendorff (2011)
    coincidence-matrix formulation. Items rated by fewer than two raters are
    dropped.
    """
    r = np.asarray(ratings, dtype=float)
    values = np.unique(r[~np.isnan(r)])
    v_idx = {v: i for i, v in enumerate(values)}
    k = len(values)
    o = np.zeros((k, k))
    for col in r.T:
        vals = col[~np.isnan(col)]
        m = len(vals)
        if m < 2:
            continue
        for p in range(m):
            for q in range(m):
                if p != q:
                    o[v_idx[vals[p]], v_idx[vals[q]]] += 1.0 / (m - 1)
    n_c = o.sum(1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")
    if level == "nominal":
        delta = 1.0 - np.eye(k)
    elif level == "interval":
        delta = (values[:, None] - values[None, :]) ** 2
    elif level == "ordinal":
        cum = np.cumsum(n_c)
        delta = np.zeros((k, k))
        for c in range(k):
            for d in range(k):
                if c != d:
                    lo, hi = min(c, d), max(c, d)
                    s = cum[hi] - (cum[lo - 1] if lo > 0 else 0) - (n_c[lo] + n_c[hi]) / 2
                    delta[c, d] = s ** 2
    else:
        raise ValueError("level must be nominal, ordinal or interval")
    d_o = (o * delta).sum() / n
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    if d_e == 0:
        return float("nan")
    return float(1 - d_o / d_e)


def bootstrap_ci(stat, a, b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0, **kw):
    """Percentile bootstrap CI for a two-rater statistic, resampling items.

    Returns (point_estimate, low, high). Pass categories=... via kw for
    kappa-type statistics so resamples keep the full category set.
    """
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    n = len(a)
    est = stat(a, b, **kw)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        v = stat(a[i], b[i], **kw)
        if not np.isnan(v):
            boots.append(v)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return est, float(lo), float(hi)


def icc(ratings, kind: str = "ICC(2,1)") -> float:
    """Intraclass correlation for an items x raters matrix (no missing values).

    kind: 'ICC(1,1)', 'ICC(2,1)' (two-way random, absolute agreement, single
    rater), 'ICC(3,1)' (two-way mixed, consistency, single rater), or the
    average-measure versions 'ICC(2,k)', 'ICC(3,k)'. Shrout & Fleiss (1979).
    """
    x = np.asarray(ratings, dtype=float)
    n, k = x.shape
    grand = x.mean()
    ss_rows = k * ((x.mean(1) - grand) ** 2).sum()
    ss_cols = n * ((x.mean(0) - grand) ** 2).sum()
    ss_tot = ((x - grand) ** 2).sum()
    ss_err = ss_tot - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    msw = (ss_cols + ss_err) / (n * (k - 1))
    if kind == "ICC(1,1)":
        return float((msr - msw) / (msr + (k - 1) * msw))
    if kind == "ICC(2,1)":
        return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))
    if kind == "ICC(3,1)":
        return float((msr - mse) / (msr + (k - 1) * mse))
    if kind == "ICC(2,k)":
        return float((msr - mse) / (msr + (msc - mse) / n))
    if kind == "ICC(3,k)":
        return float((msr - mse) / msr)
    raise ValueError("unknown ICC kind")


def confusion_table(a, b, categories=None):
    """Two-rater confusion table as a pandas DataFrame (rows: rater a)."""
    import pandas as pd
    a, b = np.asarray(a), np.asarray(b)
    if categories is None:
        categories = sorted(set(a.tolist()) | set(b.tolist()))
    m = _confusion(a, b, categories).astype(int)
    return pd.DataFrame(m, index=[f"A={c}" for c in categories],
                        columns=[f"B={c}" for c in categories])
