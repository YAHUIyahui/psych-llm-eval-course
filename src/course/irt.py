"""Two-parameter logistic (2PL) item response model, fitted by marginal
maximum likelihood with the EM algorithm (Bock & Aitkin, 1981).

P(correct | theta) = 1 / (1 + exp(-a * (theta - b)))
a = discrimination, b = difficulty. Ability prior is N(0, 1), which fixes the
scale. Missing responses are np.nan and are ignored.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def simulate_2pl(n_persons=500, n_items=30, seed=0):
    """Simulate dichotomous responses. Returns (responses, theta, a, b)."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n_persons)
    a = rng.lognormal(mean=0.0, sigma=0.3, size=n_items)
    b = rng.normal(0, 1, size=n_items)
    p = _sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
    return (rng.random(p.shape) < p).astype(float), theta, a, b


def fit_2pl(responses, n_quad: int = 41, max_iter: int = 200, tol: float = 1e-6):
    """Fit a 2PL by EM. responses: persons x items array of 0/1/nan.

    Returns dict with a, b, theta_eap, loglik, converged, n_iter.
    """
    u = np.asarray(responses, dtype=float)
    obs = ~np.isnan(u)
    u0 = np.where(obs, u, 0.0)
    n_p, n_i = u.shape

    x = np.linspace(-4, 4, n_quad)
    w = np.exp(-0.5 * x ** 2)
    w /= w.sum()

    a = np.ones(n_i)
    c = np.zeros(n_i)  # logit = a*theta + c, so b = -c / a
    prev = -np.inf
    converged = False
    for it in range(1, max_iter + 1):
        p = _sigmoid(a[None, :] * x[:, None] + c[None, :])  # Q x J
        lp, lq = np.log(p), np.log1p(-p)
        # log-likelihood of each person at each node: P x Q
        ll = u0 @ lp.T + (obs * (1 - u0)) @ lq.T
        ll += np.log(w)[None, :]
        m = ll.max(1, keepdims=True)
        post = np.exp(ll - m)
        marg = post.sum(1, keepdims=True)
        loglik = float((np.log(marg) + m).sum())
        post /= marg
        r = post.T @ u0            # Q x J expected correct
        n = post.T @ obs.astype(float)  # Q x J expected count

        for j in range(n_i):
            def nll(par, j=j):
                z = par[0] * x + par[1]
                pj = _sigmoid(z)
                f = -(r[:, j] * np.log(pj) + (n[:, j] - r[:, j]) * np.log1p(-pj)).sum()
                f += 0.5 * ((np.log(par[0]) / 0.5) ** 2) * 0.05  # weak prior on a
                g_z = -(r[:, j] - n[:, j] * pj)
                ga = (g_z * x).sum() + (np.log(par[0]) / 0.25) * 0.05 / par[0]
                return f, np.array([ga, g_z.sum()])
            res = minimize(nll, [a[j], c[j]], jac=True, method="L-BFGS-B",
                           bounds=[(0.05, 6.0), (-10, 10)])
            a[j], c[j] = res.x
        if abs(loglik - prev) < tol:
            converged = True
            break
        prev = loglik

    b = -c / a
    theta_eap = post @ x
    return {"a": a, "b": b, "theta_eap": theta_eap, "loglik": loglik,
            "converged": converged, "n_iter": it}


def item_information(theta, a, b):
    """Fisher information of each item at each theta. Shape: len(theta) x n_items."""
    theta = np.atleast_1d(theta)[:, None]
    p = _sigmoid(np.asarray(a)[None, :] * (theta - np.asarray(b)[None, :]))
    return np.asarray(a)[None, :] ** 2 * p * (1 - p)


def total_information(theta, a, b, items=None):
    """Sum of item information over `items` (all by default)."""
    info = item_information(theta, a, b)
    if items is not None:
        info = info[:, list(items)]
    return info.sum(1)


def p_correct(theta, a, b):
    """2PL probability; theta may be scalar or array, a and b arrays."""
    theta = np.atleast_1d(theta)[:, None]
    return _sigmoid(np.asarray(a)[None, :] * (theta - np.asarray(b)[None, :]))


def select_items_greedy(a, b, target_theta, n_select: int):
    """Pick `n_select` items that maximise mean test information over a grid
    of target abilities. Returns item indices in selection order."""
    chosen, remaining = [], list(range(len(a)))
    for _ in range(n_select):
        best = max(remaining, key=lambda j: total_information(target_theta, a, b, chosen + [j]).mean())
        chosen.append(best)
        remaining.remove(best)
    return chosen


def eap_theta(responses, a, b, n_quad: int = 61):
    """EAP ability estimate and posterior SD for one response vector
    (np.nan = not administered), with a N(0,1) prior."""
    x = np.linspace(-4, 4, n_quad)
    w = np.exp(-0.5 * x ** 2)
    u = np.asarray(responses, float)
    obs = ~np.isnan(u)
    p = _sigmoid(np.asarray(a)[None, obs] * (x[:, None] - np.asarray(b)[None, obs]))
    like = np.prod(np.where(u[obs][None, :] == 1, p, 1 - p), axis=1)
    post = w * like
    post /= post.sum()
    mean = float((post * x).sum())
    sd = float(np.sqrt((post * (x - mean) ** 2).sum()))
    return mean, sd


def simulate_cat(true_theta: float, a, b, se_stop: float = 0.35, max_items: int = 20, seed: int = 0):
    """Computerised adaptive test: start at theta=0, give the unused item with
    the highest information at the current estimate, update by EAP, stop when
    posterior SD < se_stop or max_items reached. Returns a list of steps."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    u = np.full(len(a), np.nan)
    est, sd = 0.0, 1.0
    steps = []
    for _ in range(max_items):
        unused = np.where(np.isnan(u))[0]
        if len(unused) == 0:
            break
        info = item_information(est, a[unused], b[unused])[0]
        j = int(unused[info.argmax()])
        p = _sigmoid(a[j] * (true_theta - b[j]))
        u[j] = float(rng.random() < p)
        est, sd = eap_theta(u, a, b)
        steps.append({"item": j, "response": int(u[j]), "theta_hat": est, "se": sd})
        if sd < se_stop:
            break
    return steps
