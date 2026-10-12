"""Print the structure of one processed perich_miller_population_2018 session.

Usage: python scripts/setup/inspect_session.py <path/to/session.h5> [...]
"""

import sys

import numpy as np
from torch_brain.data import Data, Interval


def describe_interval(name, iv):
    dur = iv.end - iv.start
    print(f"  {name}: {len(iv)} intervals, total {dur.sum():.1f} s, "
          f"len min/med/max {dur.min():.2f}/{np.median(dur):.2f}/{dur.max():.2f} s")


def main(path):
    data = Data.load(path, lazy=False)
    print("=" * 80, "\n", path)
    print(data)
    print("top-level keys:", data.keys())
    print("session:", data.session.id, "| subject:", data.subject.id)

    sp = data.spikes
    ts = sp.timestamps
    print(f"\nspikes: {len(ts)} | keys {sp.keys()} | dtype {ts.dtype}")
    print(f"  t range {ts.min():.4f} .. {ts.max():.4f} (units: seconds if range ~ session length)")
    assert np.all(np.diff(ts) >= 0), "spike timestamps not sorted"
    d = np.diff(ts)
    print(f"  smallest nonzero ISI (pooled): {d[d > 0].min():.2e} s; "
          f"timestamp grid check (ts*1e3 integer?) {np.allclose(ts * 1e3, np.round(ts * 1e3))}, "
          f"(ts*3e4 integer?) {np.allclose(ts * 3e4, np.round(ts * 3e4))}")
    print(f"  unit_index range {sp.unit_index.min()}..{sp.unit_index.max()}")

    print(f"\nunits: {len(data.units.id)} | keys {data.units.keys()}")
    print("  first ids:", list(data.units.id[:3]))
    rate = np.bincount(sp.unit_index, minlength=len(data.units.id)) / (ts.max() - ts.min())
    print(f"  firing rate Hz min/med/max {rate.min():.2f}/{np.median(rate):.2f}/{rate.max():.2f}")

    for k in data.keys():
        v = getattr(data, k)
        if hasattr(v, "timestamps") and k != "spikes":
            t = v.timestamps
            dt = np.median(np.diff(t))
            print(f"\n{k}: {len(t)} samples, keys {v.keys()}, median dt {dt * 1e3:.3f} ms "
                  f"(~{1 / dt:.1f} Hz), t {t.min():.2f}..{t.max():.2f}")

    print("\ndomains / splits:")
    for k in ("domain", "train_domain", "valid_domain", "test_domain"):
        describe_interval(k, getattr(data, k))
    tr, te = data.train_domain, data.test_domain
    assert len(tr & te) == 0 or (tr & te).end.sum() - (tr & te).start.sum() == 0, "train/test overlap"

    print("\ntrials:", len(data.trials), "| keys", data.trials.keys())
    if "is_valid" in data.trials.keys():
        print("  valid trials:", int(data.trials.is_valid.sum()))
    mp = data.movement_phases
    print("movement_phases keys:", mp.keys())
    for k in mp.keys():
        v = getattr(mp, k)
        if isinstance(v, Interval):
            describe_interval(k, v)
    for k in data.keys():
        v = getattr(data, k)
        if isinstance(v, Interval) and k not in ("domain", "train_domain", "valid_domain",
                                                   "test_domain", "trials"):
            describe_interval(k, v)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
