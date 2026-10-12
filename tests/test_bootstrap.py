"""Tests for src/bootstrap.py."""

import sys
from pathlib import Path

import numpy as np
import torch
import torchmetrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bootstrap import PairedTrialBootstrap, r2_np  # noqa: E402
from torch_brain.data import Interval  # noqa: E402


def fake_session(seed=0, n_trials=20):
    rng = np.random.default_rng(seed)
    starts = np.cumsum(rng.uniform(3, 5, n_trials))
    iv = Interval(start=starts, end=starts + 2.0)
    ts = np.concatenate([np.arange(s, s + 2.0, 0.01) for s in starts])
    target = rng.normal(size=(len(ts), 2))
    pred = target + rng.normal(scale=0.3, size=target.shape)
    return iv, ts, target, pred


def test_r2_matches_torchmetrics():
    _, _, t, p = fake_session()
    ref = torchmetrics.functional.r2_score(torch.tensor(p), torch.tensor(t)).item()
    assert abs(r2_np(p, t) - ref) < 1e-12


def test_identity_delta_is_zero():
    iv, ts, t, p = fake_session()
    b = PairedTrialBootstrap("s", ts, t, iv, p, n_boot=200)
    assert b.delta(ts, p) == (0.0, 0.0, 0.0, 0.0)


def test_matches_explicit_resampling():
    iv, ts, t, p = fake_session()
    p2 = p + np.random.default_rng(1).normal(scale=0.2, size=p.shape)
    b = PairedTrialBootstrap("s", ts, t, iv, p, n_boot=300)
    d, se, lo, hi = b.delta(ts, p2)
    trial = np.searchsorted(iv.start, ts, side="right") - 1
    groups = [np.flatnonzero(trial == k) for k in range(len(iv))]
    explicit = []
    for w in b.W.astype(int):
        idx = np.concatenate([np.repeat(groups[k], w[k]) for k in range(len(iv))])
        explicit.append(r2_np(p2[idx], t[idx]) - r2_np(p[idx], t[idx]))
    explicit = np.array(explicit)
    assert abs(d - (r2_np(p2, t) - r2_np(p, t))) < 1e-12
    np.testing.assert_allclose(np.percentile(explicit, [2.5, 97.5]), [lo, hi], atol=1e-10)
    assert abs(explicit.std(ddof=1) - se) < 1e-10


def test_rejects_mismatched_timestamps():
    iv, ts, t, p = fake_session()
    b = PairedTrialBootstrap("s", ts, t, iv, p, n_boot=50)
    try:
        b.delta(ts[:-1], p[:-1])
    except AssertionError:
        return
    raise AssertionError("mismatch not detected")
