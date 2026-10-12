"""Embedding probes: linear probes of POYO unit embeddings for unit properties, with controls.

Conventions
-----------
* Every probe is fitted on training-fold units only: a Pipeline(StandardScaler, Ridge /
  LogisticRegression) inside GridSearchCV whose inner CV is GroupKFold over the training
  sessions, so standardisation and penalty selection never see held-out units.
* Rank ("normal scores") targets: Φ⁻¹ of the training-fold empirical CDF; held-out targets
  are mapped through the training ECDF (no test information in the transform).
* Folds assign whole sessions (asserted). Scores are computed on the pooled held-out
  predictions of all folds (per held-out monkey for leave-one-monkey-out).
"""


import numpy as np
import pandas as pd
from scipy.special import ndtri
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-2, 4, 13)
INNER_FOLDS = 3


# ------------------------------------------------------------------------ transforms
def ecdf_normal_scores(train_y, y):
    """Map y through the empirical CDF of train_y (midrank, linear interpolation) to normal
    scores. Values outside the training range get the extreme training quantiles."""
    s = np.sort(np.asarray(train_y, float))
    n = len(s)
    q = (np.arange(1, n + 1) - 0.5) / n
    u = np.interp(np.asarray(y, float), s, q)
    return ndtri(np.clip(u, 0.5 / n, 1 - 0.5 / n))


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def circ_corr(a, b):
    sa = np.sin(a - np.angle(np.exp(1j * a).mean()))
    sb = np.sin(b - np.angle(np.exp(1j * b).mean()))
    return float((sa * sb).sum() / np.sqrt((sa ** 2).sum() * (sb ** 2).sum()))


def ang_err(a, b):
    return np.abs(np.angle(np.exp(1j * (a - b))))


# --------------------------------------------------------------------------- probes
def _inner_cv(groups):
    n = len(np.unique(groups))
    return GroupKFold(n_splits=min(INNER_FOLDS, n))


def fit_ridge(X, Y, groups):
    """Standardise + ridge, alpha by inner GroupKFold over training sessions."""
    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
    gs = GridSearchCV(pipe, {"ridge__alpha": ALPHAS}, cv=_inner_cv(groups),
                      scoring="neg_mean_squared_error")
    gs.fit(X, Y, groups=groups)
    return gs.best_estimator_


def check_folds(sessions, folds):
    for f in np.unique(folds):
        tr, te = set(sessions[folds != f]), set(sessions[folds == f])
        assert not (tr & te), f"session in train and test of fold {f}"


def cv_continuous(X, y, sessions, folds, rank=True):
    """Held-out predictions for one continuous target. Returns dict with arrays aligned to
    the input rows (target on the scale actually fitted, prediction, fold, alpha)."""
    check_folds(sessions, folds)
    y_t = np.full(len(y), np.nan)
    pred = np.full(len(y), np.nan)
    alpha = {}
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        ytr = ecdf_normal_scores(y[tr], y[tr]) if rank else y[tr]
        yte = ecdf_normal_scores(y[tr], y[te]) if rank else y[te]
        m = fit_ridge(X[tr], ytr, sessions[tr])
        pred[te] = m.predict(X[te])
        y_t[te] = yte
        alpha[f] = m.named_steps["ridge"].alpha
    return dict(y=y_t, pred=pred, alpha=alpha)


def cv_direction(X, theta, sessions, folds):
    """Ridge on (cos θ, sin θ); predicted angle = atan2 of the predicted pair."""
    check_folds(sessions, folds)
    pred = np.full(len(theta), np.nan)
    Y = np.column_stack([np.cos(theta), np.sin(theta)])
    alpha = {}
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        m = fit_ridge(X[tr], Y[tr], sessions[tr])
        P = m.predict(X[te])
        pred[te] = np.arctan2(P[:, 1], P[:, 0])
        alpha[f] = m.named_steps["ridge"].alpha
    return dict(y=theta, pred=pred, alpha=alpha)


# ------------------------------------------------------------- shuffled-label nulls
def null_continuous(X, y, sessions, folds, alphas, n_perm, rng, rank=True):
    """R² under permuted targets (permuted across all units), refitting per fold at the alpha
    chosen for the real target (all permutations in one multi-output ridge)."""
    P = np.stack([rng.permutation(y) for _ in range(n_perm)], 1)  # (n, n_perm)
    Yt = np.full_like(P, np.nan)
    Pr = np.full_like(P, np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if rank:
            ytr = np.column_stack([ecdf_normal_scores(P[tr, k], P[tr, k]) for k in range(n_perm)])
            yte = np.column_stack([ecdf_normal_scores(P[tr, k], P[te, k]) for k in range(n_perm)])
        else:
            ytr, yte = P[tr], P[te]
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=alphas[f]).fit(sc.transform(X[tr]), ytr)
        Pr[te] = m.predict(sc.transform(X[te]))
        Yt[te] = yte
    return np.array([r2(Yt[:, k], Pr[:, k]) for k in range(n_perm)])


def null_direction(X, theta, sessions, folds, alphas, n_perm, rng):
    out = []
    for _ in range(n_perm):
        th = rng.permutation(theta)
        Y = np.column_stack([np.cos(th), np.sin(th)])
        pred = np.full(len(th), np.nan)
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            sc = StandardScaler().fit(X[tr])
            Pf = Ridge(alpha=alphas[f]).fit(sc.transform(X[tr]), Y[tr]).predict(sc.transform(X[te]))
            pred[te] = np.arctan2(Pf[:, 1], Pf[:, 0])
        out.append(circ_corr(th, pred))
    return np.array(out)


# --------------------------------------------------------------------- baselines etc.
def session_mean_loo(y, sessions):
    """Leave-one-out mean of the unit's own session (oracle knowing only the session)."""
    df = pd.DataFrame(dict(y=y, s=sessions))
    g = df.groupby("s").y
    n, tot = g.transform("count"), g.transform("sum")
    return ((tot - df.y) / (n - 1)).where(n > 1).values


def session_circmean_loo(theta, sessions):
    """Leave-one-out circular mean direction of the unit's own session (NaN if alone)."""
    z = np.exp(1j * np.asarray(theta, float))
    df = pd.DataFrame(dict(re=z.real, im=z.imag, s=sessions))
    g = df.groupby("s")
    n = g.re.transform("count").values
    re = g.re.transform("sum").values - df.re.values
    im = g.im.transform("sum").values - df.im.values
    return np.where(n > 1, np.arctan2(im, re), np.nan)


def within_session_r2(y, pred, sessions):
    """R² after removing each session's mean from targets and predictions."""
    df = pd.DataFrame(dict(y=y, p=pred, s=sessions))
    df["y"] -= df.groupby("s").y.transform("mean")
    df["p"] -= df.groupby("s").p.transform("mean")
    return r2(df.y, df.p)


def variance_explained(X, labels):
    """Between-group share of variance per dimension and overall (sum over dims)."""
    X = np.asarray(X, float)
    mu = X.mean(0)
    tot = ((X - mu) ** 2).sum(0)
    df = pd.DataFrame(X)
    gm = df.groupby(np.asarray(labels)).transform("mean").values
    between = ((gm - mu) ** 2).sum(0)
    return between / tot, float(between.sum() / tot.sum())


def boot_sessions(fn, sessions, n_boot, rng):
    """Bootstrap over sessions: fn(index array) -> score."""
    uniq = np.unique(sessions)
    idx_by = {s: np.flatnonzero(sessions == s) for s in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        vals.append(fn(np.concatenate([idx_by[s] for s in pick])))
    return np.nanpercentile(vals, [2.5, 97.5])
