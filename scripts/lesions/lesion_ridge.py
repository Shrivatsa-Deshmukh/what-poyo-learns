"""Lesion study on CPU: binned ridge decoder refitted on the kept units only (lesion = remove).

Same sessions, conditions and lesioned sets as scripts/lesions/run_lesions.py (src/lesion.lesion_order
with the namespaced session / unit IDs; checked against the real dataset on one session).
Ridge = the ridge-control protocol (src/binned_decoder.fit_eval): 100 ms bins, K = 3 bins (300 ms
history), weighted ridge with the POYO loss weights, alpha chosen on the valid split, R² on the
test eval periods, bins on a fixed grid. Relative loss = (R²_all − R²_kept) / R²_all,
to compare with POYO's relative loss under remove.

Output: results/lesion_ridge.csv (one row per session × condition × seed × variant, plus the
all-units fit as condition "none").
Usage: python -u scripts/lesions/lesion_ridge.py --workers 6
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.binned_decoder import fit_eval, session_sets  # noqa: E402
from src.evaluate import BRAINSET, DATA_ROOT, PROJECT, TRAIN_RECORDING_IDS, SubsetPoyoMPDataset  # noqa: E402
from src.lesion import lesion_order  # noqa: E402

BIN_S = 0.1
SEEDS = (0, 1, 2)
VARIANTS = ("grid",)


def namespaced(units, s):
    u = units[units.session == s].sort_values("local_index")
    return f"{BRAINSET}/{s}", u.unit_id.tolist()


def run_session(s):
    t0 = time.perf_counter()
    units = pd.read_csv(PROJECT / "results/emb_units.csv")
    cond = pd.read_csv(PROJECT / "results/lesion_conditions.csv")
    cond = cond[cond.session == s]
    data = SubsetPoyoMPDataset(str(DATA_ROOT), [s]).get_recording(s)
    sets = session_sets(data)
    ts = np.asarray(data.spikes.timestamps, np.float64)
    un = np.asarray(data.spikes.unit_index)
    sid, ids = namespaced(units, s)
    n = len(ids)
    assert n == len(data.units.id)
    rows = []
    for v in VARIANTS:
        r = fit_eval(ts, un, n, sets, BIN_S, v)
        rows.append(dict(session=s, condition="none", seed=-1, variant=v, n_units=n, kept=n,
                         r2_test=r["r2_test"], K=r["K"]))
    for seed in SEEDS:
        order = lesion_order(ids, sid, seed)
        for c in cond.itertuples():
            les = set(order[:c.n_lesioned])
            keep = np.array([u not in les for u in ids])
            relabel = np.full(n, -1)
            relabel[keep] = np.arange(keep.sum())
            m = keep[un]
            for v in VARIANTS:
                r = fit_eval(ts[m], relabel[un[m]], int(keep.sum()), sets, BIN_S, v)
                rows.append(dict(session=s, condition=c.condition, seed=seed, variant=v,
                                 n_units=n, kept=int(keep.sum()), r2_test=r["r2_test"], K=r["K"]))
    return s, rows, round(time.perf_counter() - t0, 1)


def check_same_lesion_sets(s="c_20131003_center_out_reaching"):
    """The CSV-derived IDs give exactly the lesion order run_lesions.py uses."""
    from src.evaluate import build_dataset, load_model, nested_id
    model = load_model(device="cpu")
    _, nested = build_dataset(model, str(DATA_ROOT), [s])
    data = nested.get_recording(nested_id(s))
    units = pd.read_csv(PROJECT / "results/emb_units.csv")
    sid, ids = namespaced(units, s)
    assert sid == str(data.session.id)
    for seed in SEEDS:
        assert lesion_order(ids, sid, seed) == lesion_order(
            [str(u) for u in data.units.id], str(data.session.id), seed)
    print("lesion sets identical to run_lesions.py (checked on", s, ")", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sessions", nargs="*")
    args = ap.parse_args()
    check_same_lesion_sets()
    sessions = args.sessions or sorted(TRAIN_RECORDING_IDS)
    out = PROJECT / "results/lesion_ridge.csv"
    rows, t_start = [], time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        for s, r, dt in ex.map(run_session, sessions):
            rows += r
            pd.DataFrame(rows).to_csv(out, index=False)  # written after every session
            print(f"{s:36s} {len(r)} fits ({dt} s; {time.perf_counter() - t_start:.0f} s total)",
                  flush=True)
    print("saved", out, len(rows), "rows")


if __name__ == "__main__":
    main()
