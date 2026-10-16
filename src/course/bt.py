"""Bradley-Terry model for pairwise preferences (Hunter, 2004 MM algorithm)."""
from __future__ import annotations

import numpy as np


def fit_bradley_terry(pairs, n_items: int, max_iter: int = 1000, tol: float = 1e-10):
    """pairs: iterable of (winner, loser) index pairs; use (i, j, 0.5) style via
    two half-wins for ties. Returns log-strengths centred at 0.

    Needs a connected comparison graph in which every item wins at least once
    and loses at least once, otherwise the maximum likelihood estimate is not
    finite. A small pseudo-count of half a win each way is added to every
    compared pair for numerical safety.
    """
    wins = np.zeros((n_items, n_items))
    for w, l in pairs:
        wins[w, l] += 1.0
    n_ij = wins + wins.T
    # pseudo-count: 0.5 wins each way for every observed pair
    pc = (n_ij > 0).astype(float) * 0.5
    wins = wins + pc
    n_ij = wins + wins.T
    w_i = wins.sum(1)
    p = np.ones(n_items)
    for _ in range(max_iter):
        denom = (n_ij / (p[:, None] + p[None, :] + np.eye(n_items))).sum(1)
        new = w_i / denom
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(np.log(new) - np.log(p))) < tol:
            p = new
            break
        p = new
    return np.log(p)


def bootstrap_bt_ranks(pairs, n_items: int, n_boot: int = 500, seed: int = 0):
    """Bootstrap over comparisons. Returns (rank_samples, strength_samples).

    rank 1 = strongest. Use the spread of ranks to see which orderings are
    statistically distinguishable.
    """
    pairs = list(pairs)
    rng = np.random.default_rng(seed)
    n = len(pairs)
    ranks, strengths = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = fit_bradley_terry([pairs[i] for i in idx], n_items)
        strengths.append(s)
        ranks.append((-s).argsort().argsort() + 1)
    return np.array(ranks), np.array(strengths)


def elo_ratings(pairs, n_items: int, k: float = 16.0, base: float = 1000.0):
    """Sequential Elo ratings. Unlike Bradley-Terry, the result depends on the
    order of the comparisons."""
    r = np.full(n_items, base)
    for w, l in pairs:
        e_w = 1 / (1 + 10 ** ((r[l] - r[w]) / 400))
        r[w] += k * (1 - e_w)
        r[l] -= k * (1 - e_w)
    return r


def simulate_pairs(strength, n_comparisons: int, position_bias: float = 0.0, seed: int = 0):
    """Simulate pairwise votes. Each comparison shows a random pair in a random
    order; `position_bias` adds log-odds to whichever item is shown first.
    Returns list of (winner, loser, first_shown)."""
    rng = np.random.default_rng(seed)
    k = len(strength)
    out = []
    for _ in range(n_comparisons):
        i, j = rng.choice(k, 2, replace=False)
        logit = strength[i] - strength[j] + position_bias
        win_i = rng.random() < 1 / (1 + np.exp(-logit))
        out.append((i, j, i) if win_i else (j, i, i))
    return out
