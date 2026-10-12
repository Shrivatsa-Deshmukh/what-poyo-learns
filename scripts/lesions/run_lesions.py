"""Lesion runner: silence vs remove lesions on the POYO-MP test split (resumable).

Conditions per session come from results/lesion_conditions.csv (fractions 10/25/50/75%; kept counts
60/45/30/20 where < n_units). Both lesion types use the same lesioned units for a given session,
condition and seed (src/lesion.py). Evaluation = the timing study's path (src/evaluate.py).

Output results/<out>.csv, one row per (session, condition, lesion_type, seed):
  r2, dr2 (vs the baseline re-run in this process), n_lesioned, kept, in_training, train_cdf,
  p_unit_dropped (training UnitDropout), n_windows, n_scored, runtime_s
Usage: python -u scripts/lesions/run_lesions.py --out dev/lesion_pilot --sessions s1 s2 --seeds 0 1
       python -u scripts/lesions/run_lesions.py --out lesion --seeds 0 1 2   (all sessions)
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluate import (  # noqa: E402
    DATA_ROOT, DEFAULT_CKPT, PROJECT, TASK, TRAIN_RECORDING_IDS, build_dataset, iter_sessions,
    load_model, nested_id,
)
from src.lesion import UnitLesion, lesion_order  # noqa: E402

COLS = ["session", "task", "n_units", "condition", "kind", "fraction", "kept", "n_lesioned",
        "lesion_type", "seed", "r2", "dr2", "in_training", "train_cdf", "p_unit_dropped",
        "n_windows", "n_scored", "runtime_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = ap.parse_args()
    sessions = args.sessions or sorted(TRAIN_RECORDING_IDS)
    cond = pd.read_csv(PROJECT / "results/lesion_conditions.csv")
    sim = pd.read_csv(PROJECT / "results/lesion_dropout_sim.csv").set_index("session")
    out = PROJECT / "results" / f"{args.out}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = (pd.read_csv(out, float_precision="round_trip").to_dict("records")
            if out.exists() else [])
    done = {(r["session"], r["condition"], r["lesion_type"], int(r["seed"])) for r in rows}

    torch.manual_seed(0)
    model = load_model(DEFAULT_CKPT, "cuda")
    wrapped, nested = build_dataset(model, str(DATA_ROOT), sessions)
    t_start = time.perf_counter()

    base = {}
    for s, r in iter_sessions(model, wrapped, nested, sessions, None):
        base[s] = r["r2"]
    sweep = pd.read_csv(PROJECT / "results/sweep.csv", float_precision="round_trip")
    sb = sweep[sweep.experiment == "baseline"].set_index("session").r2
    for s in sessions:
        assert abs(base[s] - sb[s]) < 1e-6, f"{s}: baseline differs from the sweep"
    print(f"baseline done ({time.perf_counter() - t_start:.0f} s), matches sweep", flush=True)

    meta = {}
    for s in sessions:
        data = nested.get_recording(nested_id(s))
        meta[s] = (str(data.session.id), [str(u) for u in data.units.id])

    def save():
        pd.DataFrame(rows, columns=COLS).to_csv(out, index=False)

    conds = cond[cond.session.isin(sessions)]
    for seed in args.seeds:
        order = {s: lesion_order(meta[s][1], meta[s][0], seed) for s in sessions}
        for cname in conds.condition.unique():
            cc = conds[conds.condition == cname].set_index("session")
            for ltype in ("remove", "silence"):
                todo = [s for s in cc.index if (s, cname, ltype, seed) not in done]
                if not todo:
                    continue
                lesion = UnitLesion(ltype, {meta[s][0]: order[s][:int(cc.loc[s, "n_lesioned"])]
                                            for s in todo})
                for s, r in iter_sessions(model, wrapped, nested, todo, lesion):
                    c = cc.loc[s]
                    rows.append(dict(session=s, task=TASK(s), n_units=int(c.n_units),
                                     condition=cname, kind=c.kind, fraction=c.fraction,
                                     kept=int(c.kept), n_lesioned=int(c.n_lesioned),
                                     lesion_type=ltype, seed=seed, r2=r["r2"],
                                     dr2=r["r2"] - base[s], in_training=bool(c.in_training),
                                     train_cdf=float(c.train_cdf),
                                     p_unit_dropped=float(sim.loc[s, "p_unit_dropped"]),
                                     n_windows=r["n_windows"], n_scored=r["n_scored"],
                                     runtime_s=round(r["runtime_s"], 2)))
                    done.add((s, cname, ltype, seed))
                    save()
                print(f"seed {seed} {cname:7s} {ltype:7s}: {len(todo)} sessions "
                      f"({time.perf_counter() - t_start:.0f} s elapsed)", flush=True)
    print(f"saved {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
