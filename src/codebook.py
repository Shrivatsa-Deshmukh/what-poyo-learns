"""Neuron "letters": code each unit embedding with a small shared codebook, fitted on the units of
other sessions (or other monkeys), and rebuild the vector from its code.

Schemes (all pure numpy + sklearn k-means; X = unit embeddings, one row per unit):
  kmeans_quantize     one alphabet: each unit -> nearest of K centroids (K = 1 is the mean)
  product_quantize    parallel letters: PCA-rotate, split the rotated dimensions into M blocks
                      (round-robin, so the blocks carry similar variance), one K-letter alphabet
                      per block; a unit's word = M letters, its vector = the M centroids joined
                      and rotated back
  residual_quantize   coarse-to-fine letters: stage 1 = k-means on the vectors, each later stage
                      = k-means on what the earlier stages left over; vector = sum of the chosen
                      centroids
  pca_truncate        continuous control: keep the top d principal components
  group_means         area letters: each unit -> the mean of the training units of its group
  on_direction        any scheme applied to the direction (unit vector) only; each coded unit
                      keeps its own length

Bits of a code = log2(number of possible words): log2 K, M log2 K, sum of log2 K per stage.
Fits never see the units they code: the caller passes training and held-out units separately.
"""

import numpy as np
from sklearn.cluster import KMeans


def session_folds(sessions, monkeys, n_folds=5, seed=0):
    """{session: fold}. Within each monkey, sessions are shuffled and dealt to folds in turn, so
    every fold holds sessions of every monkey (that has at least n_folds sessions)."""
    sessions, monkeys = np.asarray(sessions), np.asarray(monkeys)
    assert len(sessions) == len(set(sessions)) == len(monkeys)
    rng = np.random.default_rng(seed)
    folds, start = {}, 0
    for m in sorted(set(monkeys)):
        ss = sorted(sessions[monkeys == m])
        for i, s in enumerate(rng.permutation(ss)):
            folds[str(s)] = (start + i) % n_folds
        start += len(ss)            # continue dealing where the previous monkey stopped
    return folds


def bits(scheme, k, m=1):
    """Code length in bits: k = letters per alphabet, m = number of blocks / stages."""
    assert scheme in ("kmeans", "product", "residual")
    return m * float(np.log2(k))


def fit_centroids(X, k, seed=0):
    """(k, dim) k-means centroids; k = 1 gives the mean."""
    X = np.asarray(X, dtype=np.float64)
    assert 1 <= k <= len(X), "need at least k training vectors"
    if k == 1:
        return X.mean(0, keepdims=True)
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(X)
    return km.cluster_centers_


def nearest(X, C):
    """Index of the nearest centroid (Euclidean) for each row of X; ties -> lowest index."""
    X, C = np.asarray(X, dtype=np.float64), np.asarray(C, dtype=np.float64)
    d = (X * X).sum(1)[:, None] - 2 * X @ C.T + (C * C).sum(1)[None, :]
    return d.argmin(1)


def kmeans_quantize(X_train, X_test, k, seed=0):
    """Returns (rebuilt X_test, codes (n_test,))."""
    C = fit_centroids(X_train, k, seed)
    codes = nearest(X_test, C)
    return C[codes], codes


def fit_rotation(X_train):
    """(mean (dim,), components (dim, dim)) of a full PCA; rows = components, by variance."""
    X = np.asarray(X_train, dtype=np.float64)
    mu = X.mean(0)
    _, _, vt = np.linalg.svd(X - mu, full_matrices=True)
    return mu, vt


def blocks(dim, m):
    """Round-robin split of the rotated dimensions: block j holds components j, j+m, j+2m, ..."""
    assert dim % m == 0, "dimensions must split evenly into blocks"
    return [np.arange(j, dim, m) for j in range(m)]


def product_quantize(X_train, X_test, m, k, seed=0):
    """Parallel letters. Returns (rebuilt X_test, codes (n_test, m))."""
    mu, vt = fit_rotation(X_train)
    Ztr = (np.asarray(X_train, dtype=np.float64) - mu) @ vt.T
    Zte = (np.asarray(X_test, dtype=np.float64) - mu) @ vt.T
    out = np.empty_like(Zte)
    codes = np.empty((len(Zte), m), dtype=int)
    for j, b in enumerate(blocks(Zte.shape[1], m)):
        C = fit_centroids(Ztr[:, b], k, seed + j)
        codes[:, j] = nearest(Zte[:, b], C)
        out[:, b] = C[codes[:, j]]
    return out @ vt + mu, codes


def residual_quantize(X_train, X_test, ks, seed=0):
    """Coarse-to-fine letters, one k-means per stage (ks = letters per stage).
    Returns (rebuilt X_test, codes (n_test, len(ks)))."""
    Rtr = np.asarray(X_train, dtype=np.float64).copy()
    Rte = np.asarray(X_test, dtype=np.float64).copy()
    out = np.zeros_like(Rte)
    codes = np.empty((len(Rte), len(ks)), dtype=int)
    for j, k in enumerate(ks):
        C = fit_centroids(Rtr, k, seed + j)
        Rtr -= C[nearest(Rtr, C)]
        codes[:, j] = nearest(Rte, C)
        out += C[codes[:, j]]
        Rte -= C[codes[:, j]]
    return out, codes


def pca_truncate(X_train, X_test, d):
    """Rebuild X_test from its top-d principal components (fitted on X_train)."""
    mu, vt = fit_rotation(X_train)
    V = vt[:d]
    return (np.asarray(X_test, dtype=np.float64) - mu) @ V.T @ V + mu


def group_means(X_train, g_train, g_test):
    """Each test unit -> mean of the training units of its group. Returns (vectors, codes)."""
    X = np.asarray(X_train, dtype=np.float64)
    g_train, g_test = np.asarray(g_train), np.asarray(g_test)
    labels = sorted(set(g_train))
    assert set(g_test) <= set(labels), "a test group has no training units"
    C = np.stack([X[g_train == g].mean(0) for g in labels])
    codes = np.array([labels.index(g) for g in g_test])
    return C[codes], codes


def unit_rows(X):
    """(directions (n, dim), lengths (n, 1))."""
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    assert (n > 0).all(), "zero vector has no direction"
    return X / n, n


def on_direction(quantize, X_train, X_test, *args, **kw):
    """Code only the direction of each vector with `quantize` (one of the schemes above, fitted
    on the training directions); the rebuilt direction is renormalised and given the test vector's
    own length. Returns (vectors, codes)."""
    Dtr, _ = unit_rows(X_train)
    Dte, length = unit_rows(X_test)
    v, codes = quantize(Dtr, Dte, *args, **kw)
    return v / np.linalg.norm(v, axis=1, keepdims=True) * length, codes
