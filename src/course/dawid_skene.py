"""Dawid & Skene (1979) model: each rater has a confusion matrix, the true
label of each item is latent, and EM estimates both."""
from __future__ import annotations

import numpy as np


def dawid_skene(labels, n_classes: int, max_iter: int = 100, tol: float = 1e-6, smoothing: float = 0.01):
    """labels: items x raters integer array, -1 = missing.

    Returns dict(posterior items x classes, confusion raters x K x K,
    prior K, n_iter). confusion[r, k, l] = P(rater r says l | true class k).
    """
    L = np.asarray(labels)
    n_items, n_raters = L.shape
    K = n_classes
    # initialise with per-item vote shares (soft majority vote)
    T = np.zeros((n_items, K))
    for i in range(n_items):
        for r in range(n_raters):
            if L[i, r] >= 0:
                T[i, L[i, r]] += 1
    T = (T + 1e-9) / (T + 1e-9).sum(1, keepdims=True)
    prev = -np.inf
    for it in range(1, max_iter + 1):
        prior = T.mean(0)
        conf = np.full((n_raters, K, K), smoothing)
        for r in range(n_raters):
            for i in range(n_items):
                if L[i, r] >= 0:
                    conf[r, :, L[i, r]] += T[i]
        conf /= conf.sum(2, keepdims=True)
        logT = np.tile(np.log(prior), (n_items, 1))
        for r in range(n_raters):
            mask = L[:, r] >= 0
            logT[mask] += np.log(conf[r][:, L[mask, r]]).T
        m = logT.max(1, keepdims=True)
        T = np.exp(logT - m)
        ll = float((np.log(T.sum(1)) + m[:, 0]).sum())
        T /= T.sum(1, keepdims=True)
        if abs(ll - prev) < tol:
            break
        prev = ll
    return {"posterior": T, "confusion": conf, "prior": prior, "n_iter": it}


def rater_accuracy(confusion, prior):
    """Expected accuracy of each rater under the estimated class prior."""
    return np.array([float((prior * np.diag(c)).sum()) for c in confusion])


def item_entropy(posterior):
    """Uncertainty of each item's latent label (0 = certain)."""
    p = np.clip(posterior, 1e-12, 1)
    return -(p * np.log(p)).sum(1)


def simulate_raters(n_items: int, accuracies, prior, ambiguous_share: float = 0.0, seed: int = 0):
    """Simulate labels. Each rater is correct with prob = accuracy, otherwise
    picks another class uniformly. Ambiguous items get 50% random labels for
    every rater. Returns (labels, true, ambiguous_mask)."""
    rng = np.random.default_rng(seed)
    K = len(prior)
    true = rng.choice(K, n_items, p=prior)
    amb = rng.random(n_items) < ambiguous_share
    L = np.zeros((n_items, len(accuracies)), int)
    for r, acc in enumerate(accuracies):
        for i in range(n_items):
            p_ok = 0.5 if amb[i] else acc
            if rng.random() < p_ok:
                L[i, r] = true[i]
            else:
                L[i, r] = rng.choice([k for k in range(K) if k != true[i]])
    return L, true, amb
