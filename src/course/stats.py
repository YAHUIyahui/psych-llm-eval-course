"""Interval estimation and model comparison for evaluation scores."""
from __future__ import annotations

from math import comb

import numpy as np
from scipy import stats as st


def se_proportion(p: float, n: int) -> float:
    """Standard error of a proportion."""
    return float(np.sqrt(p * (1 - p) / n))


def wald_ci(p: float, n: int, level: float = 0.95):
    z = st.norm.ppf(0.5 + level / 2)
    se = se_proportion(p, n)
    return p - z * se, p + z * se


def wilson_ci(successes: int, n: int, level: float = 0.95):
    """Wilson score interval; behaves better than Wald near 0 or 1."""
    z = st.norm.ppf(0.5 + level / 2)
    p = successes / n
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return float(centre - half), float(centre + half)


def bootstrap_ci(values, stat=np.mean, n_boot: int = 2000, level: float = 0.95, seed: int = 0):
    """Percentile bootstrap over independent units. Returns (est, low, high)."""
    v = np.asarray(values)
    rng = np.random.default_rng(seed)
    n = len(v)
    boots = np.array([stat(v[rng.integers(0, n, n)]) for _ in range(n_boot)])
    a = (1 - level) / 2
    return float(stat(v)), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def cluster_bootstrap_ci(values, clusters, stat=np.mean, n_boot: int = 2000,
                         level: float = 0.95, seed: int = 0):
    """Bootstrap that resamples whole clusters (e.g. conversations), keeping
    the observations inside a cluster together."""
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    ids = np.unique(c)
    groups = {g: v[c == g] for g in ids}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        boots.append(stat(np.concatenate([groups[g] for g in pick])))
    a = (1 - level) / 2
    return float(stat(v)), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def hedges_g(x, y):
    """Standardised mean difference with small-sample correction (x minus y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    d = (x.mean() - y.mean()) / sp
    j = 1 - 3 / (4 * (nx + ny) - 9)
    return float(d * j)


def paired_compare(a, b, n_boot: int = 4000, seed: int = 0):
    """Compare two systems scored on the same items.

    Returns dict with mean difference (a - b), bootstrap 95% CI over items,
    paired t-test p value, and the unpaired CI for contrast.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    est, lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    t = st.ttest_rel(a, b)
    se_unpaired = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return {"diff": est, "ci_low": lo, "ci_high": hi, "p_paired_t": float(t.pvalue),
            "unpaired_ci": (est - 1.96 * se_unpaired, est + 1.96 * se_unpaired)}


def mcnemar_exact(a, b):
    """Exact McNemar test for paired binary outcomes (1 = pass).

    Uses only discordant items: b01 (a fails, b passes), b10 (a passes, b fails).
    """
    a, b = np.asarray(a).astype(int), np.asarray(b).astype(int)
    b10 = int(((a == 1) & (b == 0)).sum())
    b01 = int(((a == 0) & (b == 1)).sum())
    n = b10 + b01
    if n == 0:
        return {"b10": 0, "b01": 0, "p": 1.0}
    k = min(b10, b01)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"b10": b10, "b01": b01, "p": float(p)}


def p_adjust(pvals, method: str = "holm"):
    """Multiple-comparison adjustment: 'bonferroni', 'holm' or 'bh'."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    if method == "bonferroni":
        return np.minimum(1, p * m)
    if method == "holm":
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, (m - rank) * p[i])
            adj[i] = min(1.0, running)
        return adj
    if method == "bh":
        running = 1.0
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            running = min(running, p[i] * m / (rank + 1))
            adj[i] = running
        return adj
    raise ValueError("method must be bonferroni, holm or bh")


def required_items(sd_diff: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Items needed to detect a mean paired difference `delta`, given the SD of
    per-item differences (normal approximation, two-sided)."""
    z_a = st.norm.ppf(1 - alpha / 2)
    z_b = st.norm.ppf(power)
    return int(np.ceil(((z_a + z_b) * sd_diff / delta) ** 2))


def simulate_power(n_items: int, p_a: float, p_b: float, rho: float = 0.5,
                   alpha: float = 0.05, n_sim: int = 2000, seed: int = 0) -> float:
    """Monte-Carlo power of the exact McNemar test for two correlated binary
    systems with pass rates p_a and p_b and latent correlation rho."""
    rng = np.random.default_rng(seed)
    za, zb = st.norm.ppf(p_a), st.norm.ppf(p_b)
    hits = 0
    for _ in range(n_sim):
        z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n_items)
        a = (z[:, 0] < za).astype(int)
        b = (z[:, 1] < zb).astype(int)
        hits += mcnemar_exact(a, b)["p"] < alpha
    return hits / n_sim
