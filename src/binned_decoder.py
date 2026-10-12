"""Control decoder: ridge regression on binned spike counts (no POYO).

Per session and bin size b: features = spike counts per unit in K = ceil(0.25 s / b) bins of
width b (>= 250 ms of history), target = cursor velocity. Bins lie on a fixed absolute grid
[k*b, (k+1)*b), as in the snap / bin-shuffle perturbations; the most recent bin is the one
containing t, so it looks up to ~b/2 into the future on average (POYO sees the whole 1 s window).

Protocol (same as the POYO evaluation, examples/poyo/datasets/poyo_mp.py:165-193):
  train  = cursor samples in train_domain with POYO loss weight > 0, weighted by it
           (examples/poyo/datasets/wrapper.py:_resolve_weights)
  valid  = cursor samples in valid_domain ∩ eval_interval (alpha selection)
  test   = cursor samples in test_domain ∩ eval_interval; R² = mean over x/y (as
           torchmetrics r2_score). Unique timestamps (no overlapping-window duplicates).
Ridge: features z-scored with weighted train statistics, intercept unpenalised, alpha
chosen on valid from trace(A)/p * logspace(-7, 2, 19).
"""

import numpy as np
import scipy.sparse as sp

from src.bootstrap import r2_np

ALPHA_GRID = np.logspace(-7, 2, 19)


def edges_for(t, b, K, variant):
    if variant != "grid":
        raise ValueError(variant)
    top = (np.floor(np.round(t * 1e6) / round(b * 1e6)) + 1) * b  # end of bin holding t
    return top[:, None] - b * np.arange(K + 1)[None, :]  # (n, K+1), descending


def count_features(spike_ts, spike_unit, n_units, t, b, K, variant):
    """Sparse (n, n_units*K) matrix of spike counts; column block k = lag k."""
    E = edges_for(t, b, K, variant)
    key = np.round(E * 1e9).astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(E.shape)
    uniq_t = uniq / 1e9
    ncum = np.empty((len(uniq), n_units), np.int32)
    for u in range(n_units):  # spikes strictly before each edge
        ncum[:, u] = np.searchsorted(spike_ts[spike_unit == u], uniq_t, side="left")
    blocks = [sp.csr_matrix(ncum[inv[:, k - 1]] - ncum[inv[:, k]]) for k in range(1, K + 1)]
    X = sp.hstack(blocks, format="csr").astype(np.float64)
    assert X.min() >= 0
    return X


class WeightedRidge:
    def fit(self, X, y, w):
        sw = w.sum()
        self.mu = np.asarray(X.T @ w).ravel() / sw
        # dense BLAS in row chunks: same result as the sparse product, ~3x faster
        XtWX = np.zeros((X.shape[1], X.shape[1]))
        for i in range(0, X.shape[0], 20000):
            Xc = X[i:i + 20000].toarray()
            XtWX += (Xc * w[i:i + 20000, None]).T @ Xc
        XtWX -= sw * np.outer(self.mu, self.mu)
        var = np.diag(XtWX) / sw
        self.sd = np.where(var > 1e-12, np.sqrt(np.clip(var, 0, None)), 1.0)
        A = XtWX / np.outer(self.sd, self.sd)
        self.ybar = (w[:, None] * y).sum(0) / sw
        XtWy = (X.T @ (w[:, None] * (y - self.ybar))) / self.sd[:, None]
        self.evals, self.evecs = np.linalg.eigh(A)
        self.proj = self.evecs.T @ XtWy
        self.scale = np.trace(A) / A.shape[0]
        return self

    def coef(self, alpha):
        lam = alpha * self.scale
        beta = self.evecs @ (self.proj / (np.clip(self.evals, 0, None) + lam)[:, None])
        return beta / self.sd[:, None]  # back to raw-count units

    def predict(self, X, beta):
        return np.asarray(X @ beta) - self.mu @ beta + self.ybar


def fit_eval(spike_ts, spike_unit, n_units, sets, b, variant):
    """sets: dict(train=(t, y, w), valid=(t, y), test=(t, y)). Returns dict of results."""
    K = int(np.ceil(0.25 / b - 1e-9))
    feats = {k: count_features(spike_ts, spike_unit, n_units, v[0], b, K, variant)
             for k, v in sets.items()}
    t_tr, y_tr, w_tr = sets["train"]
    model = WeightedRidge().fit(feats["train"], y_tr, w_tr)
    val = []
    for a in ALPHA_GRID:
        beta = model.coef(a)
        val.append(r2_np(model.predict(feats["valid"], beta), sets["valid"][1]))
    best = int(np.argmax(val))
    # lower edge = (near-)unregularised least squares, fine if valid R² is flat there;
    # upper edge would mean the grid is too narrow -> fail loudly
    assert best < len(ALPHA_GRID) - 1, "alpha at upper grid edge"
    beta = model.coef(ALPHA_GRID[best])
    r2_test = r2_np(model.predict(feats["test"], beta), sets["test"][1])
    return dict(K=K, history_ms=1000 * K * b, n_features=feats["train"].shape[1],
                alpha_idx=best, alpha_at_edge=best in (0, len(ALPHA_GRID) - 1),
                r2_valid_spread_low_alphas=float(np.ptp(val[:4])),
                r2_valid=val[best], r2_test=r2_test)


def session_sets(data):
    """Train / valid / test cursor samples of one session, as described above.
    data: a recording from SubsetPoyoMPDataset (has readout_config)."""
    # imported here: src.evaluate puts the POYO example (datasets.wrapper) on sys.path
    import src.evaluate  # noqa: F401
    from datasets.wrapper import _resolve_weights
    from torch_brain.utils import isin_interval

    cfg = data.readout_config
    t = np.asarray(data.cursor.timestamps, np.float64)
    y = np.asarray(data.get_nested_attribute(cfg.value_key), np.float64)
    w = _resolve_weights(t, data, weight_cfg=cfg.weights).astype(np.float64)
    ev = data.get_nested_attribute(cfg.eval_interval)
    tr = isin_interval(t, data.train_domain) & (w > 0)
    va = isin_interval(t, data.valid_domain) & isin_interval(t, ev)
    te = isin_interval(t, data.test_domain) & isin_interval(t, ev)
    assert tr.sum() > 1000 and va.sum() > 50 and te.sum() > 50
    assert not np.any(tr & te) and not np.any(tr & va)
    return dict(train=(t[tr], y[tr], w[tr]), valid=(t[va], y[va]), test=(t[te], y[te]))
