"""Tests for src/unit_properties.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.unit_properties import (  # noqa: E402
    BIN_S, LAGS, N_FOLDS, _folds, cv_r2_all_lags, firing_stats, scored_bins, tuning,
)
from torch_brain.data import Interval  # noqa: E402


def smooth_velocity(n, rng):
    """Random smooth 2-D velocity (~200 ms correlation), shape (n, 2)."""
    w = rng.normal(size=(n + 50, 2))
    k = np.exp(-0.5 * (np.arange(-25, 26) / 8) ** 2)
    return np.stack([np.convolve(w[:, d], k, mode="valid")[:n] for d in range(2)], 1) * 5


def synthetic(n=20000, lag_bins=5, pd_rad=1.0, depth=2.0, b0=20.0, noise=3.0, seed=0):
    rng = np.random.default_rng(seed)
    vel = smooth_velocity(n, rng)
    j = np.arange(40, n - 40)
    v = vel[j + lag_bins]
    tuned = b0 + depth * (np.cos(pd_rad) * v[:, 0] + np.sin(pd_rad) * v[:, 1])
    flat = np.full(len(j), b0)
    Y = np.stack([tuned, flat], 1) + rng.normal(scale=noise, size=(len(j), 2))
    return Y, vel, j, rng


def test_recovers_lag_direction_depth():
    Y, vel, j, rng = synthetic(lag_bins=5, pd_rad=1.0, depth=2.0)
    r = tuning(Y, vel, j, n_shifts=20, rng=rng)
    assert r["best_lag_ms"][0] == pytest.approx(5 * BIN_S * 1000)
    assert abs(np.angle(np.exp(1j * (np.arctan2(r["by"][0], r["bx"][0]) - 1.0)))) < 0.05
    assert np.hypot(r["bx"][0], r["by"][0]) == pytest.approx(2.0, rel=0.05)
    assert r["tuning_r2"][0] > 0.5
    # untuned unit: R² ~ 0 and not above its null threshold; tuned unit above
    thr = np.percentile(r["null"], 95, axis=0)
    assert r["tuning_r2"][0] > thr[0]
    assert r["tuning_r2"][1] < 0.01 and r["tuning_r2"][1] <= thr[1] + 1e-3


def test_negative_lag_recovered():
    Y, vel, j, rng = synthetic(lag_bins=-8, seed=3)
    r = tuning(Y, vel, j, n_shifts=0, rng=rng)
    assert r["best_lag_ms"][0] == pytest.approx(-8 * BIN_S * 1000)


def test_cv_r2_matches_explicit_refit():
    Y, vel, j, _ = synthetic(n=3000, seed=1)
    fast = cv_r2_all_lags(Y, vel, j)
    for li in (0, 7, len(LAGS) - 1):
        L = LAGS[li]
        X = np.stack([np.ones(len(j)), vel[j + L, 0], vel[j + L, 1]], 1)
        pred = np.empty_like(Y)
        for b in _folds(len(j)):
            tr = np.setdiff1d(np.arange(len(j)), b)
            beta = np.linalg.lstsq(X[tr], Y[tr], rcond=None)[0]
            pred[b] = X[b] @ beta
        r2 = 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
        np.testing.assert_allclose(fast[li], r2, atol=1e-9)
    assert N_FOLDS == 5


def test_null_masking_consistent():
    """With a shift that puts invalid (NaN) velocity bins under some scored rows, the CV R² must
    equal an explicit refit that drops those rows."""
    Y, vel, j, _ = synthetic(n=3000, seed=2)
    vel = vel.copy()
    vel[1000:1100] = np.nan
    shift = 400
    li = list(LAGS).index(0)
    fast = cv_r2_all_lags(Y, vel, j, shift=shift)[li]
    v = np.roll(vel, shift, axis=0)[j]
    ok = ~np.isnan(v).any(1)
    X = np.stack([np.ones(len(j)), np.nan_to_num(v[:, 0]), np.nan_to_num(v[:, 1])], 1)
    pred = np.empty_like(Y)
    for b in _folds(len(j)):
        tr = np.setdiff1d(np.arange(len(j)), b)
        tr = tr[ok[tr]]
        beta = np.linalg.lstsq(X[tr], Y[tr], rcond=None)[0]
        pred[b] = X[b] @ beta
    ybar = Y[ok].mean(0)
    r2 = 1 - ((Y - pred) ** 2 * ok[:, None]).sum(0) / ((Y - ybar) ** 2 * ok[:, None]).sum(0)
    np.testing.assert_allclose(fast, r2, atol=1e-9)


def test_scored_bins_respect_forbidden_and_lags():
    n = 1000
    edges = np.arange(n + 1) * BIN_S
    vel = np.zeros((n, 2))
    vel[500] = np.nan
    iv = Interval(start=np.array([2.0, 12.0]), end=np.array([4.0, 15.0]))
    j = scored_bins(edges, iv, vel)
    c = (edges[:-1] + edges[1:]) / 2
    assert np.all(((c[j] >= 2) & (c[j] < 4)) | ((c[j] >= 12) & (c[j] < 15)))
    for L in LAGS:
        assert not np.isnan(vel[j + L]).any()
    forb = Interval(start=np.array([4.1]), end=np.array([5.0]))  # within +300 ms of 2-4 s
    with pytest.raises(AssertionError):
        scored_bins(edges, iv, vel, forbidden=forb)
    far = Interval(start=np.array([7.0]), end=np.array([8.0]))
    scored_bins(edges, iv, vel, forbidden=far)


def test_firing_stats_isi_within_intervals_only():
    iv = Interval(start=np.array([0.0, 10.0]), end=np.array([1.0, 11.0]))
    ts = np.array([0.1, 0.2, 0.3, 10.1, 10.2, 10.3, 5.0])   # 5.0 is outside
    un = np.zeros(len(ts), int)
    f = firing_stats(ts, un, 1, iv)
    assert f["spike_count"][0] == 6 and f["rate_hz"][0] == pytest.approx(3.0)
    assert f["isi_cv"][0] == pytest.approx(0.0, abs=1e-9)  # ISIs 0.1 only; 9.8 s gap excluded


def test_coarse_r2_higher_than_fine_for_noisy_tuned_unit():
    from src.unit_properties import coarse_cv_r2
    Y, vel, j, rng = synthetic(n=20000, lag_bins=5, noise=150.0, seed=4)  # 20 ms R² low
    r = tuning(Y, vel, j, n_shifts=0, rng=rng)
    assert r["tuning_r2"][0] < 0.2
    rate = np.zeros((len(vel), 2))
    rate[j] = Y
    r100, n100 = coarse_cv_r2(rate, vel, j, np.array([5, 0]))
    assert n100 > 0.95 * len(j) / 5
    assert r100[0] > 2 * r["tuning_r2"][0]   # averaging 5 bins removes much of the noise
    assert abs(r100[1]) < 0.01                # untuned unit stays at ~0


def test_coarse_r2_matches_manual_computation():
    from src.unit_properties import _folds, coarse_cv_r2
    Y, vel, j, _ = synthetic(n=6000, lag_bins=3, noise=40.0, seed=5)
    rate = np.zeros((len(vel), 2))
    rate[j] = Y
    r100, _ = coarse_cv_r2(rate, vel, j, np.array([3, 3]))
    js = set(j.tolist())
    k = np.array([kk for kk in np.unique(j // 5) if all(5 * kk + i in js for i in range(5))])
    y = np.stack([rate[5 * kk:5 * kk + 5].mean(0) for kk in k])
    v = np.stack([vel[5 * kk + 3:5 * kk + 8].mean(0) for kk in k])
    X = np.column_stack([np.ones(len(k)), v])
    pred = np.empty_like(y)
    for b in _folds(len(k)):
        tr = np.setdiff1d(np.arange(len(k)), b)
        pred[b] = X[b] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
    manual = 1 - ((y - pred) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)
    np.testing.assert_allclose(r100, manual, atol=1e-12)


def test_in_intervals_matches_library_and_moving_intervals():
    from src.unit_properties import in_intervals, moving_intervals
    from torch_brain.utils import isin_interval
    rng = np.random.default_rng(6)
    s = np.sort(rng.choice(np.arange(0, 1000, 0.5), 300, replace=False))
    iv = Interval(start=s, end=s + 0.3)
    t = rng.uniform(-5, 1005, 20000)
    assert np.array_equal(in_intervals(t, iv), isin_interval(t, iv))
    n = 5000
    edges = np.arange(n + 1) * BIN_S
    vel = smooth_velocity(n, rng)
    vel[100:120] = np.nan
    ref = Interval(start=np.array([10.0]), end=np.array([60.0]))
    mov, thr = moving_intervals(edges, vel, ref)
    c = (edges[:-1] + edges[1:]) / 2
    sp = np.hypot(vel[:, 0], vel[:, 1])
    expect = ~np.isnan(sp) & (sp > thr)
    assert np.array_equal(in_intervals(c, mov), expect)
    r = in_intervals(c, ref) & ~np.isnan(sp)
    assert abs((sp[r] > thr).mean() - 0.8) < 0.01
