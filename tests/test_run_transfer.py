"""Tests for the donor choice and condition vectors of scripts/transfer/run_transfer.py
(no model needed: random embedding tables with the real unit table)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts" / "transfer"))

UNITS = PROJECT / "results" / "emb_units.csv"
pytestmark = pytest.mark.skipif(not UNITS.exists(), reason="results/emb_units.csv missing")


@pytest.fixture(scope="module")
def env():
    import run_transfer as rt
    units = pd.read_csv(UNITS)
    rng = np.random.default_rng(0)
    W = rng.normal(size=(units.emb_row.max() + 1, 8))
    sessions = sorted(units.session.unique())
    srow = {s: i + 1 for i, s in enumerate(sessions)}
    Ws = rng.normal(size=(len(sessions) + 1, 8))
    meta = units.drop_duplicates("session").set_index("session")
    return rt, units, W, Ws, srow, meta


SESSIONS = ["m_20140203_center_out_reaching", "c_20131009_random_target_reaching",
            "j_20160405_center_out_reaching"]


@pytest.mark.parametrize("s", SESSIONS)
def test_donors_respect_monkey_task_and_gap_rules(env, s):
    rt, units, _, _, _, meta = env
    d = rt.donors_for(s, units)
    assert d == rt.donors_for(s, units)                                     # deterministic
    m, t = meta.loc[s, "monkey"], meta.loc[s, "task"]
    same_task = [x for x in meta.index if x != s and meta.loc[x, "monkey"] == m
                 and meta.loc[x, "task"] == t]
    assert d["main"] in same_task
    assert rt.days(s, d["main"]) == min(rt.days(s, x) for x in same_task)  # nearest date
    assert meta.loc[d["other_monkey"], "monkey"] != m and meta.loc[d["other_monkey"], "task"] == t
    bins = {name: (lo, hi) for lo, hi, name in rt.GAPS}
    for name, g in d["gaps"].items():
        lo, hi = bins[name]
        assert g in same_task and lo <= rt.days(s, g) <= hi


@pytest.mark.parametrize("s", SESSIONS)
def test_condition_vectors(env, s):
    rt, units, W, Ws, srow, _ = env
    tgt = units[units.session == s]
    rows = set(tgt.emb_row)
    conds = {(c, dn, seed): (cov, ur, sr) for c, dn, _, seed, cov, ur, sr
             in rt.conditions(s, units, W, Ws, srow)}
    for (c, _, _), (_, ur, sr) in conds.items():
        if ur is not None:
            assert set(ur) == rows and 0 not in ur                       # only the target's rows
        if sr is not None:
            assert list(sr) == [srow[s]]                                 # only the target's session
    _, own, own_s = conds[("own_swap", None, -1)]
    assert all(np.array_equal(own[r], W[r]) for r in rows)
    assert np.array_equal(own_s[srow[s]], Ws[srow[s]])

    # own-electrode mean: unchanged for units alone on their electrode
    _, oem, _ = conds[("own_electrode_mean", None, -1)]
    alone = tgt.groupby("channel").filter(lambda g: len(g) == 1)
    assert len(alone) and all(np.allclose(oem[r], W[r]) for r in alone.emb_row)

    # matched: donor electrode means, donor mean as fallback, coverage = matched fraction
    donor = rt.donors_for(s, units)["main"]
    dn = units[units.session == donor]
    cov, ur, _ = conds[("matched", donor, -1)]
    shared = tgt.channel.isin(set(dn.channel))
    assert cov == pytest.approx(shared.mean())
    for r, c in zip(tgt.emb_row, tgt.channel):
        want = (W[dn[dn.channel == c].emb_row].mean(0) if c in set(dn.channel)
                else W[dn.emb_row].mean(0))
        assert np.allclose(ur[r], want)
    for seed in (0, 1, 2):  # same coverage when shuffled
        assert conds[("shuffled_within_area", donor, seed)][0] == pytest.approx(cov)
        # within-area shuffle: every matched unit gets an electrode mean from its own area
        _, ua, _ = conds[("shuffled_within_area", donor, seed)]
        area_means = {a: [W[g.emb_row].mean(0) for _, g in dn[dn.area == a].groupby("channel")]
                      for a in dn.area.unique()}
        for r, c, a in zip(tgt.emb_row, tgt.channel, tgt.area):
            if c in set(dn.channel):
                assert any(np.allclose(ua[r], v) for v in area_means[a])
    om = rt.donors_for(s, units)["other_monkey"]
    _, _, sm = conds[("session_other_monkey", om, -1)]
    assert np.array_equal(sm[srow[s]], Ws[srow[om]])


def test_ridge_apply_imputes_missing_donor_electrodes_with_training_mean():
    import ridge_cross_day as rc
    rng = np.random.default_rng(0)
    channels = ["a", "b", "c"]
    n = len(channels) * rc.K
    dec = dict(channels=channels, mu=rng.normal(size=n), beta=rng.normal(size=(n, 2)),
               ybar=np.array([0.5, -0.5]))
    t = np.arange(1.0, 3.0, 0.1)
    ts = np.sort(rng.uniform(0, 3, 300))
    ch = rng.choice(["a", "c", "z"], size=300)              # "b" absent on this day, "z" unknown
    pred, missing = rc.apply(dec, ts, ch, t, {"a", "c", "z"})
    assert missing == pytest.approx(1 / 3)
    X = rc.features(ts, ch, channels, t)
    cols_b = [k * 3 + 1 for k in range(rc.K)]
    assert not X[:, cols_b].any()                            # no spikes ever reach "b"
    X[:, cols_b] = dec["mu"][cols_b]
    want = X @ dec["beta"] - dec["mu"] @ dec["beta"] + dec["ybar"]
    assert np.allclose(pred, want)
    # an imputed electrode contributes exactly nothing beyond the mean
    dec0 = dict(dec, beta=dec["beta"].copy())
    dec0["beta"][cols_b] = 0
    assert np.allclose(rc.apply(dec0, ts, ch, t, {"a", "c", "z"})[0], pred)
