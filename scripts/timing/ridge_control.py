"""Binned ridge decoder control on the POYO-MP sessions (resumable).

Writes results/binned_ridge.csv: session, task, variant, bin_ms, K, history_ms, n_features,
alpha_idx, alpha_at_edge, r2_valid, r2_test, n_train, n_test, runtime_s.

Usage: python scripts/timing/ridge_control.py [--sessions ...]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.binned_decoder import fit_eval, session_sets  # noqa: E402
from src.evaluate import (  # noqa: E402
    BRAINSET, DATA_ROOT, PROJECT, TASK, TRAIN_RECORDING_IDS, SubsetPoyoMPDataset,
)

BINS_MS = [10, 20, 50, 100, 150, 200, 333, 500]
VARIANTS = ["grid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--out", default="binned_ridge")
    args = ap.parse_args()
    sessions = args.sessions or sorted(
        s for s in TRAIN_RECORDING_IDS if (DATA_ROOT / BRAINSET / f"{s}.h5").exists())
    out = PROJECT / "results" / f"{args.out}.csv"
    rows = pd.read_csv(out, float_precision="round_trip").to_dict("records") if out.exists() else []
    done = {(r["session"], r["variant"], float(r["bin_ms"])) for r in rows}
    ds = SubsetPoyoMPDataset(str(DATA_ROOT), sessions)
    for s in sessions:
        todo = [(v, b) for v in VARIANTS for b in BINS_MS if (s, v, float(b)) not in done]
        if not todo:
            continue
        data = ds.get_recording(s)  # adds readout_config (example's get_recording_hook)
        sets = session_sets(data)
        ts = np.asarray(data.spikes.timestamps, np.float64)
        un = np.asarray(data.spikes.unit_index)
        n_units = len(data.units.id)
        for v, b in todo:
            t0 = time.perf_counter()
            r = fit_eval(ts, un, n_units, sets, b / 1000, v)
            rows.append(dict(session=s, task=TASK(s), variant=v, bin_ms=float(b), **r,
                             n_train=len(sets["train"][0]), n_test=len(sets["test"][0]),
                             runtime_s=round(time.perf_counter() - t0, 2)))
            pd.DataFrame(rows).to_csv(out, index=False)
        last = [r for r in rows if r["session"] == s]
        print(f"{s:36s} " + " ".join(f"{r['variant'][0]}{r['bin_ms']:g}={r['r2_test']:.3f}"
                                     for r in last), flush=True)
    print(f"saved {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
