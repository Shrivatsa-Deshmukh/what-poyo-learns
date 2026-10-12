"""Invariant tests for src/perturb.py (snap, within-bin shuffle, window shuffle)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.perturb import (  # noqa: E402
    SpikeTimingPerturbation, _bin_ns, _to_ns, shuffle_within_bin, shuffle_within_window,
    snap_to_bin_center,
)

BINS_S = [0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100]
SESSION_H5 = (PROJECT / "data/processed/perich_miller_population_2018/"
              "c_20131003_center_out_reaching.h5")
CKPT = PROJECT / "checkpoints/poyo_mp.ckpt"


# ------------------------------------------------------------------ synthetic data
def synthetic_window(seed=0, n=2000, n_units=30, offset=123.456789, length=1.0):
    """Spikes on a 30 kHz grid in absolute time, returned window-relative."""
    rng = np.random.default_rng(seed)
    first = int(np.ceil(offset * 30000))
    last = int(np.ceil((offset + length) * 30000)) - 1
    ticks = np.sort(rng.integers(first, last + 1, size=n))
    t_abs = ticks / 30000.0
    ts = t_abs - offset
    units = rng.integers(0, n_units, size=n)
    keep = (ts >= 0) & (ts < length)
    return ts[keep], units[keep], 0.0, length, offset


def abs_bin(ts, offset, bin_s):
    return _to_ns(ts + offset) // _bin_ns(bin_s)


def counts(units, bins):
    pairs = np.stack([units, bins], 1)
    vals, c = np.unique(pairs, axis=0, return_counts=True)
    return {tuple(v): k for v, k in zip(vals, c)}


def run(mode, ts, units, b, t0, t1, off, seed=0):
    if mode == "snap":
        return snap_to_bin_center(ts, units, b, t0, t1, off)
    return shuffle_within_bin(ts, units, b, t0, t1, off, seed=seed)


# ------------------------------------------------------------- pure-function tests
@pytest.mark.parametrize("mode", ["snap", "bin_shuffle"])
@pytest.mark.parametrize("b", BINS_S)
def test_e1_e2_counts_sorted_bounds_units(mode, b):
    ts, units, t0, t1, off = synthetic_window()
    new, new_u, order = run(mode, ts, units, b, t0, t1, off)
    assert np.all(np.diff(new) >= 0), "not sorted"
    assert np.all((new >= t0) & (new < t1)), "left the window"
    assert np.array_equal(new_u, units[order])
    assert np.array_equal(np.sort(new_u), np.sort(units)), "unit multiset changed"
    assert counts(units, abs_bin(ts, off, b)) == counts(new_u, abs_bin(new, off, b)), \
        "per-(unit, bin) counts changed"


@pytest.mark.parametrize("b", BINS_S)
def test_e1_interior_bins_match_formula(b):
    """Away from window edges, snap is exactly floor(T/b)*b + b/2 in absolute time."""
    ts, units, t0, t1, off = synthetic_window()
    new, _, order = snap_to_bin_center(ts, units, b, t0, t1, off)
    T = ts[order] + off
    expected = np.floor(_to_ns(T) / _bin_ns(b)) * b + b / 2
    k = abs_bin(ts[order], off, b)
    interior = (k * b - off >= t0) & ((k + 1) * b - off <= t1)
    assert interior.sum() > 0
    np.testing.assert_allclose(new[interior] + off, expected[interior], atol=1e-9)


def test_window_edge_clipping():
    """Bins cut by the window edges: snap -> center of the clipped part, bin shuffle -> inside it."""
    b = 0.100
    off = 10.03  # window [10.03, 11.03): bin [10.0, 10.1) is cut at 10.03, [11.0, 11.1) at 11.03
    ts = np.array([0.0, 0.01, 0.065, 0.5, 0.975, 0.99, 0.9999])
    units = np.arange(len(ts))
    new, nu, order = snap_to_bin_center(ts, units, b, 0.0, 1.0, off)
    got = dict(zip(nu, new))
    # first bin clipped to [0, 0.07): center 0.035
    for u in (0, 1, 2):
        assert got[u] == pytest.approx(0.035, abs=1e-12)
    assert got[3] == pytest.approx(0.52, abs=1e-12)  # bin [10.5, 10.6) -> 10.55 -> 0.52
    # last bin [11.0, 11.1) clipped to [0.97, 1.0): center 0.985
    for u in (4, 5, 6):
        assert got[u] == pytest.approx(0.985, abs=1e-12)
    for seed in range(20):
        new2, nu2, _ = shuffle_within_bin(ts, units, b, 0.0, 1.0, off, seed=seed)
        g2 = dict(zip(nu2, new2))
        assert all(0.0 <= g2[u] < 0.07 for u in (0, 1, 2))
        assert all(0.97 <= g2[u] < 1.0 for u in (4, 5, 6))
        assert np.all((new2 >= 0) & (new2 < 1.0))


@pytest.mark.parametrize("mode", ["snap", "bin_shuffle"])
@pytest.mark.parametrize("b", [0.005, 0.010, 0.050])
def test_overlapping_windows_consistent(mode, b):
    """A spike whose bin is fully inside both overlapping windows gets the same new
    absolute time in both; spikes in bins cut by an edge stay inside their window."""
    rng = np.random.default_rng(1)
    S = 200.012345
    ticks = np.sort(rng.integers(int(np.ceil(S * 3e4)), int((S + 1.5) * 3e4), size=4000))
    T = ticks / 3e4
    U = rng.integers(0, 40, size=len(T))
    res = {}
    for w0 in (S, S + 0.5):
        m = (T >= w0) & (T < w0 + 1.0)
        new, nu, order = run(mode, T[m] - w0, U[m], b, 0.0, 1.0, w0, seed=3)
        assert np.all((new >= 0) & (new < 1.0))
        res[w0] = dict(zip(zip(nu, ticks[m][order]), new + w0))
    shared = set(res[S]) & set(res[S + 0.5])
    bn = _bin_ns(b)
    lo_ns, hi_ns = _to_ns(S + 0.5), _to_ns(S + 1.0)
    full = [key for key in shared
            if (key[1] * 1e9 / 3e4 // bn) * bn >= lo_ns
            and (key[1] * 1e9 / 3e4 // bn + 1) * bn <= hi_ns]
    assert len(full) > 100
    for key in full:
        assert res[S][key] == pytest.approx(res[S + 0.5][key], abs=1e-9)


def test_e2_determinism_and_seed_dependence():
    ts, units, t0, t1, off = synthetic_window()
    a = shuffle_within_bin(ts, units, 0.01, t0, t1, off, seed=7)[0]
    b = shuffle_within_bin(ts, units, 0.01, t0, t1, off, seed=7)[0]
    c = shuffle_within_bin(ts, units, 0.01, t0, t1, off, seed=8)[0]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_e4_counts_bounds_determinism():
    ts, units, t0, t1, off = synthetic_window()
    new, nu, _ = shuffle_within_window(ts, units, t0, t1, np.random.default_rng(5))
    assert np.all(np.diff(new) >= 0) and np.all((new >= t0) & (new < t1))
    assert np.array_equal(np.bincount(nu, minlength=30), np.bincount(units, minlength=30))
    again = shuffle_within_window(ts, units, t0, t1, np.random.default_rng(5))[0]
    assert np.array_equal(new, again)
    assert not np.array_equal(new, np.sort(ts)), "window shuffle left times unchanged"


def test_rejects_spikes_outside_window():
    with pytest.raises(AssertionError):
        snap_to_bin_center(np.array([0.5, 1.2]), np.array([0, 1]), 0.01, 0.0, 1.0)


# ---------------------------------------------------------- transform on real data
needs_data = pytest.mark.skipif(not SESSION_H5.exists(), reason="session not prepared")


def real_window(start_offset=0.0):
    from torch_brain.data import Data
    data = Data.load(SESSION_H5, lazy=False)
    S = float(data.test_domain.start[3]) + start_offset
    return data.slice(S, S + 1.0), S


def same_value(a, b):
    from torch_brain.data import Data, Interval, IrregularTimeSeries
    from torch_brain.data.arraydict import ArrayDict
    if isinstance(a, (Data, IrregularTimeSeries, Interval, ArrayDict)):
        ka, kb = sorted(a.__dict__), sorted(b.__dict__)
        return ka == kb and all(same_value(a.__dict__[k], b.__dict__[k]) for k in ka)
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b)
    return a == b


ALL_PERTURBATIONS = ([SpikeTimingPerturbation("snap", b) for b in BINS_S]
                     + [SpikeTimingPerturbation("bin_shuffle", b, seed=s)
                        for b in BINS_S for s in range(3)]
                     + [SpikeTimingPerturbation("window_shuffle", seed=s) for s in range(3)])


@needs_data
@pytest.mark.parametrize("tf", ALL_PERTURBATIONS, ids=repr)
def test_transform_only_spike_times_change(tf):
    win, S = real_window()
    ref, _ = real_window()  # independent slice (deepcopy of a sliced Data raises in torch_brain 0.2.0)
    out = tf(win)
    assert len(ref.spikes) > 100
    # non-spike fields are untouched
    for key in ref.keys():
        if key != "spikes":
            assert same_value(getattr(out, key), getattr(ref, key)), key
    assert same_value(out._domain, ref._domain) and out.absolute_start == ref.absolute_start
    sp, rs = out.spikes, ref.spikes
    assert sorted(sp.keys()) == sorted(rs.keys())
    assert np.all(np.diff(sp.timestamps) >= 0)
    assert np.all((sp.timestamps >= 0) & (sp.timestamps < out.domain.end[0]))
    n_units = len(ref.units.id)
    assert np.array_equal(np.bincount(sp.unit_index, minlength=n_units),
                          np.bincount(rs.unit_index, minlength=n_units))
    if tf.mode in ("snap", "bin_shuffle"):
        assert counts(rs.unit_index, abs_bin(rs.timestamps, S, tf.bin_s)) == \
            counts(sp.unit_index, abs_bin(sp.timestamps, S, tf.bin_s))
    assert not np.array_equal(sp.timestamps, rs.timestamps), "nothing was perturbed"


@needs_data
@pytest.mark.parametrize("tf", [SpikeTimingPerturbation("none"),
                                SpikeTimingPerturbation("snap", 0.0),
                                SpikeTimingPerturbation("bin_shuffle", 0.0, seed=1)], ids=repr)
def test_transform_identity(tf):
    win, _ = real_window()
    ref, _ = real_window()
    out = tf(win)
    assert out is win and same_value(out, ref)


@needs_data
@pytest.mark.parametrize("tf", ALL_PERTURBATIONS[::4], ids=repr)
def test_transform_deterministic(tf):
    a = tf(real_window()[0]).spikes.timestamps
    b = tf(real_window()[0]).spikes.timestamps
    assert np.array_equal(a, b)


# ------------------------------------------------- tokenizer / model-level invariants
needs_model = pytest.mark.skipif(not (SESSION_H5.exists() and CKPT.exists()),
                                 reason="checkpoint or data missing")


@pytest.fixture(scope="module")
def model_and_window():
    sys.path.insert(0, str(PROJECT))
    from src.evaluate import DATA_ROOT, build_dataset, load_model, nested_id
    from torch_brain.data import Data
    model = load_model(CKPT, "cpu")
    _, nested = build_dataset(model, str(DATA_ROOT),
                              ["c_20131003_center_out_reaching"])
    rid = nested_id("c_20131003_center_out_reaching")
    S = float(Data.load(SESSION_H5, lazy=False).test_domain.start[3])

    def window():
        from torch_brain.datasets import DatasetIndex
        return nested[DatasetIndex(rid, S, S + 1.0)]
    return model, window


@needs_model
@pytest.mark.parametrize("tf", [SpikeTimingPerturbation("snap", 0.05),
                                SpikeTimingPerturbation("bin_shuffle", 0.01, seed=2),
                                SpikeTimingPerturbation("window_shuffle", seed=0)], ids=repr)
def test_delimiter_and_latent_tokens_unchanged(model_and_window, tf):
    model, window = model_and_window
    a = model.tokenize(window())
    b = model.tokenize(tf(window()))
    for k in ("latent_index", "latent_timestamps", "session_index"):
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k
    ta, tb = (np.asarray(x["input_token_type"].obj if hasattr(x["input_token_type"], "obj")
                         else x["input_token_type"]) for x in (a, b))
    assert np.array_equal(ta, tb)
    get = lambda d, k: np.asarray(d[k].obj if hasattr(d[k], "obj") else d[k])  # noqa: E731
    delim = ta != 0
    assert delim.sum() == 2 * len(window().units.id)
    for k in ("input_unit_index", "input_timestamps"):
        assert np.array_equal(get(a, k)[delim], get(b, k)[delim]), k
    assert not np.array_equal(get(a, "input_timestamps"), get(b, "input_timestamps"))


@needs_model
@pytest.mark.parametrize("tf", [SpikeTimingPerturbation("none"),
                                SpikeTimingPerturbation("snap", 0.0),
                                SpikeTimingPerturbation("bin_shuffle", 0.0)], ids=repr)
def test_model_output_bit_identical_at_identity(model_and_window, tf):
    from torch_brain.batching import collate
    model, window = model_and_window
    out_ts = torch.linspace(0, 0.99, 50)[None]

    def predict(sample):
        X = collate([model.tokenize(sample)])
        with torch.no_grad():
            return model(**X, output_timestamps=out_ts)
    assert torch.equal(predict(window()), predict(tf(window())))
