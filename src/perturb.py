"""Test-time spike-timing perturbations for POYO: snap, bin shuffle, window shuffle.

Conventions
-----------
* Pure functions take window-relative spike times ``ts`` (seconds, as produced by
  ``Data.slice``), ``units`` (unit index per spike), the window bounds
  ``[t_start, t_end)`` (relative), and ``offset`` = the window's absolute start time
  (``data.absolute_start``), so that ``ts + offset`` is absolute recording time.
* Bins are anchored to absolute recording time: bin k = [k*b, (k+1)*b). Bin indices are
  computed on an integer-nanosecond grid so that the same spike gets the same bin in every
  window that contains it (float round-off from ``ts + offset`` is ~1e-13 s << 0.5 ns).
* Window edges: a bin cut by a window edge is clipped to the part inside the window.
  Snap moves spikes to the center of the clipped part, bin shuffle draws uniformly inside
  it. Moved spikes therefore never leave the window. Eval windows lie inside their sampling interval
  (SequentialFixedWindowSampler), so they never leave the interval either.
* Bin-shuffle randomness is a counter-based hash of (seed, unit id, absolute spike time in ns), not
  a sequential RNG. A spike whose bin lies entirely inside two overlapping windows gets the
  same new absolute time in both. Spikes in a bin cut by an edge of one of the windows
  can differ between the windows (the clipped bin differs), which is unavoidable once
  spikes must stay inside each window.
* Window shuffle is per window by definition: overlapping windows get independent shuffles,
  seeded by (seed, session id, window absolute start) so results do not depend on loader
  order.
* Outputs are re-sorted by time (stable sort); all per-spike fields move together.
"""

import copy
import hashlib

import numpy as np

from torch_brain.data import Data, IrregularTimeSeries

_NS = 1_000_000_000
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


# ----------------------------------------------------------------------------- helpers
def _to_ns(t_abs):
    return np.round(np.asarray(t_abs, dtype=np.float64) * _NS).astype(np.int64)


def _bin_ns(bin_s):
    b = int(round(bin_s * _NS))
    assert b > 0 and abs(b - bin_s * _NS) < 1e-3, f"bin {bin_s} s not representable in ns"
    return b


def clipped_bins(ts, bin_s, t_start, t_end, offset):
    """Absolute bin index per spike and the [lo, hi) bounds (window-relative) of that bin
    clipped to the window."""
    ts = np.asarray(ts, dtype=np.float64)
    b = _bin_ns(bin_s)
    k = _to_ns(ts + offset) // b
    lo = np.maximum(k * b / _NS - offset, t_start)
    hi = np.minimum((k + 1) * b / _NS - offset, t_end)
    # bin membership is decided on the integer-ns grid; a spike exactly on a bin's lower
    # edge can sit ~1 ulp below the float `lo`. Allow < 1 ns, fail loudly beyond that.
    assert np.all(lo - ts < 1e-9), "spike below its own bin by more than 1 ns"
    lo = np.minimum(lo, ts)
    assert np.all(ts < hi), "spike at/above the top of its own clipped bin"
    return k, lo, hi


def _splitmix64(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & _MASK64
    x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _MASK64
    x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _MASK64
    return x ^ (x >> np.uint64(31))


def stable_key(s) -> int:
    """64-bit key of a string (stable across runs, unlike Python's hash)."""
    if isinstance(s, bytes):
        s = s.decode()
    return int.from_bytes(hashlib.blake2b(str(s).encode(), digest_size=8).digest(), "little")


def hash_uniform(seed, unit_keys, t_ns):
    """Deterministic U[0, 1) per (seed, unit key, absolute time in ns)."""
    with np.errstate(over="ignore"):
        x = _splitmix64(np.uint64(seed) ^ _splitmix64(np.asarray(unit_keys, np.uint64)))
        x = _splitmix64(x ^ np.asarray(t_ns, np.int64).astype(np.uint64))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _check_inputs(ts, units, t_start, t_end):
    ts = np.asarray(ts, dtype=np.float64)
    units = np.asarray(units)
    assert ts.ndim == 1 and ts.shape == units.shape
    assert t_end > t_start
    assert np.all((ts >= t_start) & (ts < t_end)), "input spikes outside window"
    return ts, units


def _sorted(ts_new, units):
    order = np.argsort(ts_new, kind="stable")
    return ts_new[order], units[order], order


# ------------------------------------------------------------------ pure perturbations
def snap_to_bin_center(ts, units, bin_s, t_start, t_end, offset=0.0):
    """Snap: move each spike to the center of its (window-clipped) absolute bin.
    Returns (ts_new, units_new, order) with order such that ts_new = moved[order]."""
    ts, units = _check_inputs(ts, units, t_start, t_end)
    _, lo, hi = clipped_bins(ts, bin_s, t_start, t_end, offset)
    return _sorted((lo + hi) / 2.0, units)


def shuffle_within_bin(ts, units, bin_s, t_start, t_end, offset=0.0, seed=0, unit_keys=None):
    """Bin shuffle: move each spike to a uniform random time inside its (window-clipped) absolute
    bin. Randomness = hash(seed, unit key, absolute spike time) so a spike gets the same
    draw in every window. unit_keys[u] = 64-bit key of unit u (default: u itself)."""
    ts, units = _check_inputs(ts, units, t_start, t_end)
    _, lo, hi = clipped_bins(ts, bin_s, t_start, t_end, offset)
    keys = units if unit_keys is None else np.asarray(unit_keys, np.uint64)[units]
    u = hash_uniform(seed, keys, _to_ns(ts + offset))
    new = lo + u * (hi - lo)
    new = np.where(new < hi, new, np.nextafter(hi, lo))  # guard round-off at the top edge
    return _sorted(new, units)


def shuffle_within_window(ts, units, t_start, t_end, rng):
    """Window shuffle: replace every spike time by a uniform draw over [t_start, t_end); each
    unit keeps its spike count in the window."""
    ts, units = _check_inputs(ts, units, t_start, t_end)
    new = rng.uniform(t_start, t_end, size=len(ts))
    new = np.where(new < t_end, new, np.nextafter(t_end, t_start))
    return _sorted(new, units)


# ---------------------------------------------------------------------- transform
class SpikeTimingPerturbation:
    r"""Transform that perturbs spike times of a sliced window (style of
    :class:`torch_brain.transforms.UnitDropout` / ``RandomTimeScaling``: a callable that
    takes and returns a :class:`Data`). Apply after slicing and before ``model.tokenize``.

    Args:
        mode: "snap", "bin_shuffle", "window_shuffle" or "none".
        bin_s: bin size in seconds (snap, bin_shuffle). ``bin_s == 0`` means identity.
        seed: seed for bin_shuffle / window_shuffle.
        field: spike field, default "spikes".
    """

    MODES = ("none", "snap", "bin_shuffle", "window_shuffle")

    def __init__(self, mode: str, bin_s: float = 0.0, seed: int = 0, field: str = "spikes"):
        assert mode in self.MODES, mode
        if mode in ("snap", "bin_shuffle"):
            assert bin_s >= 0
        self.mode, self.bin_s, self.seed, self.field = mode, bin_s, seed, field

    @property
    def is_identity(self):
        return self.mode == "none" or (self.mode in ("snap", "bin_shuffle") and self.bin_s == 0)

    def __call__(self, data: Data) -> Data:
        if self.is_identity:
            return data
        spikes = getattr(data, self.field)
        assert isinstance(spikes, IrregularTimeSeries)
        dom = data.domain
        assert len(dom) == 1, "expected a single contiguous window"
        t0, t1 = float(dom.start[0]), float(dom.end[0])
        offset = float(data.absolute_start)
        ts, units = spikes.timestamps, spikes.unit_index

        if self.mode == "snap":
            new_ts, _, order = snap_to_bin_center(ts, units, self.bin_s, t0, t1, offset)
        elif self.mode == "bin_shuffle":
            keys = np.array([stable_key(u) for u in data.units.id], dtype=np.uint64)
            new_ts, _, order = shuffle_within_bin(
                ts, units, self.bin_s, t0, t1, offset, self.seed, keys)
        else:
            rng = np.random.default_rng(
                [self.seed, stable_key(data.session.id), int(_to_ns(offset))])
            new_ts, _, order = shuffle_within_window(ts, units, t0, t1, rng)

        out = copy.copy(spikes)  # shallow, as in transforms/random_time_scaling.py
        for key in spikes.keys():
            if key != "timestamps":
                setattr(out, key, getattr(spikes, key)[order])
        out.timestamps = new_ts
        assert out.is_sorted()
        setattr(data, self.field, out)
        return data

    def __repr__(self):
        return f"SpikeTimingPerturbation(mode={self.mode!r}, bin_s={self.bin_s}, seed={self.seed})"
