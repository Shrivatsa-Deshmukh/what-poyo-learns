"""Unit properties: per-unit firing and velocity-tuning properties from data (no model).

Intervals ("used"): chosen by the caller (scripts/embeddings/unit_properties.py): the training
split ∩ moving_intervals(), i.e. 20 ms bins faster than the 20th percentile of the speed in the
training split ∩ the readout eval interval (used_intervals(..., "scored"); CO:
movement_phases.reach_period, RT: movement_phases.random_period;
examples/poyo/datasets/poyo_mp.py:179,192), and the whole training split for firing statistics.

Firing: spikes inside the used intervals; rate = count / total duration; ISI CV from ISIs
between consecutive spikes of a unit inside the same interval.

Velocity tuning (linear/cosine): the whole recording is binned on an absolute 20 ms grid
(spike counts -> rate in Hz; cursor velocity = mean of the 100 Hz samples in the bin; bins
overlapping cursor_outlier_segments have no velocity). Model, per lag L (−300 … +300 ms, 20 ms
steps; positive lag = neural activity leads movement):
    rate(t) = b0 + bx·vx(t + L) + by·vy(t + L)
fitted by OLS, all units at once (shared design matrix). Scored samples = bins whose centre is
in the used intervals and whose velocity bin t + L is valid for every lag (one common sample
set, so lags are comparable). 5-fold CV over contiguous blocks of these samples; CV R² =
1 − Σ held-out SSE / Σ (y − ȳ)². Best lag = max CV R². Coefficients from a full fit at the
best lag: tuning depth √(bx² + by²) (Hz per velocity unit), depth / mean rate, preferred
direction atan2(by, bx). Optional speed model adds b_s·|v| (CV R² reported separately).

Null: velocity circularly shifted by a random offset in [5 s, T − 5 s] (100 shifts), with the
full lag search repeated inside each shift; well_tuned = CV R² above the unit's 95th percentile
of null (max-over-lag) CV R².
"""

import numpy as np

from torch_brain.data import Interval

BIN_S = 0.02
LAGS = np.arange(-15, 16)  # bins: −300 … +300 ms
N_FOLDS = 5


# ----------------------------------------------------------------------------- intervals
def used_intervals(data, split, scope="all"):
    """scope="all": the whole <split>_domain (*_whole columns; the main
    scope is this ∩ moving_intervals(), built by the caller); scope="scored": <split>_domain ∩
    the readout eval interval (reach / random-target periods; secondary *_scored columns).
    Bins without valid velocity are dropped later by scored_bins()."""
    dom = getattr(data, f"{split}_domain")
    dom = Interval(start=np.asarray(dom.start), end=np.asarray(dom.end))  # start/end only
    if scope == "all":
        return dom
    assert scope == "scored", scope
    ev = data.get_nested_attribute(data.readout_config.eval_interval)
    return dom & ev


def in_intervals(t, iv):
    """Boolean mask: t in [start, end) of any interval of a sorted, disjoint Interval. Same
    result as torch_brain.utils.isin_interval (utils/weights.py:6-24) but O(n log m) memory-free,
    needed for scopes made of thousands of short segments."""
    s, e = np.asarray(iv.start, np.float64), np.asarray(iv.end, np.float64)
    if len(s) == 0:
        return np.zeros(len(t), bool)
    assert np.all(np.diff(s) >= 0) and np.all(s[1:] >= e[:-1]), "intervals must be sorted, disjoint"
    t = np.asarray(t, np.float64)
    k = np.searchsorted(s, t, side="right") - 1
    ok = k >= 0
    ok[ok] = t[ok] < e[k[ok]]
    return ok


def moving_intervals(edges, vel, ref_iv, pct=20.0):
    """Intervals of 20 ms bins whose speed exceeds the `pct` percentile of the speed of bins
    whose centre lies in ref_iv (train split ∩ scored periods; valid velocity only).
    Returns (Interval of merged runs of moving bins, threshold)."""
    speed = np.hypot(vel[:, 0], vel[:, 1])
    centres = (edges[:-1] + edges[1:]) / 2
    valid = ~np.isnan(speed)
    ref = in_intervals(centres, ref_iv) & valid
    assert ref.sum() > 100, "too few reference bins for the speed threshold"
    thr = float(np.percentile(speed[ref], pct))
    mov = valid & (speed > thr)
    d = np.diff(np.concatenate([[0], mov.astype(int), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return Interval(start=edges[starts], end=edges[ends]), thr


def total_duration(iv):
    return float(np.sum(iv.end - iv.start))


# ------------------------------------------------------------------------------- firing
def firing_stats(spike_ts, spike_unit, n_units, iv):
    """count, rate (Hz), log10 rate (NaN if 0), ISI CV (NaN if < 3 spikes) per unit."""
    inside = in_intervals(spike_ts, iv)
    ts, un = spike_ts[inside], spike_unit[inside]
    seg = np.searchsorted(iv.start, ts, side="right") - 1
    count = np.bincount(un, minlength=n_units)
    rate = count / total_duration(iv)
    cv = np.full(n_units, np.nan)
    for u in range(n_units):
        m = un == u
        if m.sum() < 3:
            continue
        t, s = ts[m], seg[m]
        order = np.argsort(t, kind="stable")
        t, s = t[order], s[order]
        isi = np.diff(t)[np.diff(s) == 0]  # only within the same interval
        if len(isi) >= 2 and isi.mean() > 0:
            cv[u] = isi.std() / isi.mean()
    with np.errstate(divide="ignore"):
        logr = np.where(rate > 0, np.log10(np.where(rate > 0, rate, 1)), np.nan)
    return dict(spike_count=count, rate_hz=rate, log10_rate=logr, isi_cv=cv, inside=inside)


# ------------------------------------------------------------------------------- binning
def bin_recording(data):
    """Absolute 20 ms grid over the recording domain: rate (n_bins, n_units) float32,
    velocity (n_bins, 2) with NaN where invalid, bin edges."""
    t_end = float(np.max(data.domain.end))
    n_bins = int(np.ceil(t_end / BIN_S))
    edges = np.arange(n_bins + 1) * BIN_S
    st = np.asarray(data.spikes.timestamps)
    su = np.asarray(data.spikes.unit_index)
    n_units = len(data.units.id)
    k = np.floor(st / BIN_S).astype(np.int64)
    ok = (k >= 0) & (k < n_bins)
    counts = np.zeros((n_bins, n_units), np.float32)
    np.add.at(counts, (k[ok], su[ok]), 1.0)
    rate = counts / BIN_S

    ct = np.asarray(data.cursor.timestamps)
    cv = np.asarray(data.cursor.vel, np.float64)
    kc = np.floor(ct / BIN_S).astype(np.int64)
    okc = (kc >= 0) & (kc < n_bins)
    n = np.bincount(kc[okc], minlength=n_bins)
    vel = np.full((n_bins, 2), np.nan)
    for d in range(2):
        s = np.bincount(kc[okc], weights=cv[okc, d], minlength=n_bins)
        vel[:, d] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    # bins overlapping a cursor outlier segment have no valid velocity
    out = data.cursor_outlier_segments
    for a, b in zip(out.start, out.end):
        vel[int(np.floor(a / BIN_S)):int(np.ceil(b / BIN_S))] = np.nan
    return rate, vel, edges


def scored_bins(edges, iv, vel, forbidden=None):
    """Bin indices j whose centre is in iv and whose velocity at j + L is valid for all lags.
    `forbidden` (Interval): no scored bin and no velocity bin may overlap it (asserted)."""
    centres = (edges[:-1] + edges[1:]) / 2
    j = np.flatnonzero(in_intervals(centres, iv))
    n_bins = len(centres)
    valid_v = ~np.isnan(vel).any(1)
    keep = np.ones(len(j), bool)
    for L in LAGS:
        jl = j + L
        inside = (jl >= 0) & (jl < n_bins)
        keep &= inside
        keep[inside] &= valid_v[jl[inside]]
    j = j[keep]
    if forbidden is not None and len(forbidden):
        used = np.unique(np.concatenate([j + L for L in LAGS]))
        bl, bh = edges[:-1][used], edges[1:][used]
        ov = np.zeros(len(used), bool)
        for a, b in zip(forbidden.start, forbidden.end):
            ov |= (bl < b) & (bh > a)
        assert not ov.any(), "a scored or velocity bin overlaps a forbidden (held-out) interval"
    return j


# ------------------------------------------------------------------------- tuning fits
def _folds(n):
    return np.array_split(np.arange(n), N_FOLDS)


def cv_r2_all_lags(Y, vel, j, extra_speed=False, shift=0):
    """CV R² (n_lags, n_units) for the linear velocity model at every lag.
    Y: (n, U) rates at scored bins j. vel: (n_bins, 2). shift: circular shift of velocity
    (bins) for the null. Uses per-block sufficient statistics: for block k,
    SSE_k = Σy² − 2 bᵀXᵀy + bᵀXᵀX b with b fitted on the other blocks."""
    n, U = Y.shape
    blocks = _folds(n)
    v = np.roll(vel, shift, axis=0) if shift else vel
    out = np.empty((len(LAGS), U))
    for li, L in enumerate(LAGS):
        V = v[j + L]
        ok = ~np.isnan(V).any(1)  # always all True for shift = 0 (scored_bins guarantees it)
        assert shift or ok.all()
        V = np.where(ok[:, None], V, 0.0)
        cols = [np.ones(n), V[:, 0], V[:, 1]]
        if extra_speed:
            cols.append(np.hypot(V[:, 0], V[:, 1]))
        X = np.stack(cols, 1) * ok[:, None]   # rows with invalid velocity drop out of
        Yw = Y * ok[:, None]                  # every statistic (X, y, Σy², SST)
        ybar = Yw.sum(0) / ok.sum()
        sst = ((Y - ybar) ** 2 * ok[:, None]).sum(0)
        yy_k = np.stack([(Yw[b] ** 2).sum(0) for b in blocks])     # (K, U)
        XtX_k = np.stack([X[b].T @ X[b] for b in blocks])          # (K, p, p)
        XtY_k = np.stack([X[b].T @ Yw[b] for b in blocks])         # (K, p, U)
        XtX, XtY = XtX_k.sum(0), XtY_k.sum(0)
        sse = np.zeros(U)
        for k in range(len(blocks)):
            beta = np.linalg.solve(XtX - XtX_k[k], XtY - XtY_k[k])   # (p, U)
            sse += (yy_k[k] - 2 * np.einsum("pu,pu->u", beta, XtY_k[k])
                    + np.einsum("pu,pq,qu->u", beta, XtX_k[k], beta))
        out[li] = 1 - sse / np.where(sst > 0, sst, np.nan)
    return out


def full_fit(Y, vel, j, lag_bins):
    """OLS coefficients (3, U) at each unit's own best lag."""
    U = Y.shape[1]
    beta = np.empty((3, U))
    for L in np.unique(lag_bins):
        us = np.flatnonzero(lag_bins == L)
        V = vel[j + L]
        X = np.stack([np.ones(len(j)), V[:, 0], V[:, 1]], 1)
        beta[:, us] = np.linalg.lstsq(X, Y[:, us], rcond=None)[0]
    return beta


def _max_over_lags(r2):
    """Max over lags; NaN for units whose R² is undefined (no spikes in the scored bins)."""
    m = np.where(np.isnan(r2), -np.inf, r2).max(0)
    return np.where(np.isinf(m), np.nan, m)


def tuning(Y, vel, j, n_shifts=100, rng=None, min_shift_s=5.0):
    """Best lag, CV R², coefficients, speed-model R², null threshold and p-value per unit."""
    r2 = cv_r2_all_lags(Y, vel, j)
    best = np.nanargmax(np.where(np.isnan(r2), -np.inf, r2), axis=0)
    r2_best = r2[best, np.arange(Y.shape[1])]
    lag_bins = LAGS[best]
    beta = full_fit(Y, vel, j, lag_bins)
    r2_speed = np.empty(Y.shape[1])
    for L in np.unique(lag_bins):
        us = np.flatnonzero(lag_bins == L)
        sel = np.flatnonzero(LAGS == L)[0]
        r2_speed[us] = cv_r2_all_lags(Y[:, us], vel, j, extra_speed=True)[sel]
    null = None
    if n_shifts:
        n_bins = len(vel)
        lo = int(round(min_shift_s / BIN_S))
        assert n_bins - 2 * lo > 0, "recording too short for the shift null"
        shifts = rng.integers(lo, n_bins - lo, size=n_shifts)
        null = np.stack([_max_over_lags(cv_r2_all_lags(Y, vel, j, shift=int(s)))
                         for s in shifts])  # (n_shifts, U): lag search inside the null
    return dict(r2=r2, best_lag_ms=lag_bins * BIN_S * 1000, tuning_r2=r2_best,
                b0=beta[0], bx=beta[1], by=beta[2], speed_model_r2=r2_speed, null=null)


COARSE = 5  # 100 ms bins = 5 consecutive 20 ms bins


def coarse_cv_r2(rate, vel, j, lag_bins, factor=COARSE):
    """CV R² at 100 ms resolution, at each unit's lag chosen on the 20 ms fit (no new lag
    search). A 100 ms bin k covers 20 ms bins [factor·k, factor·k + factor); it is used if all
    its sub-bins are scored bins. Rate = mean of the sub-bin rates; velocity for lag L = mean of
    the 20 ms velocity bins shifted by L (so the lag keeps 20 ms precision). 5-fold CV over
    contiguous blocks, as at 20 ms."""
    js = set(j.tolist())
    k = np.unique(j // factor)
    k = k[[all((factor * kk + i) in js for i in range(factor)) for kk in k]]
    sub = factor * k[:, None] + np.arange(factor)[None, :]           # (n_k, factor)
    Y = rate[sub].mean(1)                                            # (n_k, U)
    out = np.full(rate.shape[1], np.nan)
    blocks = _folds(len(k))
    for L in np.unique(lag_bins):
        us = np.flatnonzero(lag_bins == L)
        V = vel[sub + L].mean(1)                                     # (n_k, 2)
        assert not np.isnan(V).any()
        X = np.stack([np.ones(len(k)), V[:, 0], V[:, 1]], 1)
        y = Y[:, us]
        pred = np.empty_like(y)
        for b in blocks:
            tr = np.setdiff1d(np.arange(len(k)), b)
            pred[b] = X[b] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        sst = ((y - y.mean(0)) ** 2).sum(0)
        out[us] = 1 - ((y - pred) ** 2).sum(0) / np.where(sst > 0, sst, np.nan)
    return out, len(k)
