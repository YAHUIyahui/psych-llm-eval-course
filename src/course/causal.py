"""Simulations and estimators for causal questions about chatbot use."""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_chatbot_use(n: int = 4000, true_effect: float = -2.0, randomised: bool = False, seed: int = 0):
    """Older adults; outcome = loneliness score after 8 weeks (higher = lonelier).

    Observational version: lonelier people and people with higher digital
    literacy are more likely to use the chatbot (self-selection), and baseline
    loneliness also drives the outcome, so naive comparisons are confounded.
    """
    rng = np.random.default_rng(seed)
    age = rng.normal(70, 6, n)
    literacy = rng.normal(0, 1, n) - 0.05 * (age - 70)
    lonely0 = rng.normal(40, 8, n)
    if randomised:
        use = rng.random(n) < 0.5
    else:
        logit = -0.5 + 0.08 * (lonely0 - 40) + 0.9 * literacy
        use = rng.random(n) < 1 / (1 + np.exp(-logit))
    lonely1 = 8 + 0.8 * lonely0 + 0.3 * (age - 70) + true_effect * use + rng.normal(0, 4, n)
    return pd.DataFrame({"age": age, "literacy": literacy, "lonely0": lonely0,
                         "use": use.astype(int), "lonely1": lonely1})


def naive_difference(df, y="lonely1", t="use"):
    return float(df.loc[df[t] == 1, y].mean() - df.loc[df[t] == 0, y].mean())


def regression_adjusted(df, covariates, y="lonely1", t="use"):
    X = np.column_stack([np.ones(len(df)), df[t].values] + [df[c].values for c in covariates])
    beta, *_ = np.linalg.lstsq(X, df[y].values, rcond=None)
    return float(beta[1])


def ipw_ate(df, covariates, y="lonely1", t="use"):
    """Inverse probability weighting with a logistic propensity model."""
    from sklearn.linear_model import LogisticRegression
    X = df[covariates].values
    ps = LogisticRegression(max_iter=1000).fit(X, df[t]).predict_proba(X)[:, 1]
    ps = np.clip(ps, 0.02, 0.98)
    T, Y = df[t].values, df[y].values
    return float(np.mean(T * Y / ps) - np.mean((1 - T) * Y / (1 - ps)))


def did_estimate(pre_t, post_t, pre_c, post_c):
    """Difference-in-differences from four group means."""
    return float((post_t - pre_t) - (post_c - pre_c))
