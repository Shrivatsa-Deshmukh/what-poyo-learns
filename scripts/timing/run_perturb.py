"""Run spike-timing perturbation experiments on the POYO-MP test split (resumable).

Experiments (src/perturb.py):
  snap            move each spike to the centre of its time bin (bin sizes from --bins-ms)
  bin_shuffle     move each spike to a random time inside its bin (seeds from --seeds)
  window_shuffle  move each spike to a random time inside its 1 s window (no bin)
  identity        sanity check: transform inserted with bin 0, must equal the baseline

Writes results/<out>.csv, one row per (experiment, session, setting, seed):
  experiment, session, task, perturbation, bin_ms, seed, r2, n_windows, n_scored, runtime_s,
  dr2, dr2_boot_se, dr2_ci_lo, dr2_ci_hi
dr2* = paired trial-bootstrap of R²(perturbed) − R²(baseline) for that session
(src/bootstrap.py, 1000 resamples of test trials, same resamples for every setting).

Resume: rows already in the CSV (same experiment/perturbation/bin_ms/seed/session) are
skipped. The baseline is always re-run (its predictions are needed for the paired CIs) and
must reproduce any stored baseline R² exactly.

Usage:
  python scripts/timing/run_perturb.py --out dev/pilot --sessions s1 s2 \
      --experiments baseline snap bin_shuffle window_shuffle
  python scripts/timing/run_perturb.py --out sweep          # all prepared POYO-MP sessions
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.bootstrap import PairedTrialBootstrap  # noqa: E402
from src.evaluate import (  # noqa: E402
    BRAINSET, DATA_ROOT, DEFAULT_CKPT, PROJECT, TASK, TRAIN_RECORDING_IDS, build_dataset,
    iter_sessions, load_model,
)
from src.perturb import SpikeTimingPerturbation  # noqa: E402

ALL_BINS_MS = [1, 2, 5, 10, 20, 50, 75, 100, 150, 200, 333, 500]
COLUMNS = ["experiment", "session", "task", "perturbation", "bin_ms", "seed", "r2",
           "n_windows", "n_scored", "runtime_s", "dr2", "dr2_boot_se", "dr2_ci_lo",
           "dr2_ci_hi"]


def settings(experiments, bins_ms, seeds):
    """Yield (experiment, perturbation_name, bin_ms, seed, transform or None)."""
    for exp in experiments:
        if exp == "baseline":
            continue  # always run first, see main()
        elif exp == "identity":  # transform inserted, but bin = 0 -> must equal baseline
            yield exp, "snap", 0.0, -1, SpikeTimingPerturbation("snap", 0.0)
            yield exp, "bin_shuffle", 0.0, 0, SpikeTimingPerturbation("bin_shuffle", 0.0)
        elif exp == "snap":
            for b in bins_ms:
                yield exp, "snap", float(b), -1, SpikeTimingPerturbation("snap", b / 1000)
        elif exp == "bin_shuffle":
            for b in bins_ms:
                for s in seeds:
                    yield exp, "bin_shuffle", float(b), s, SpikeTimingPerturbation(
                        "bin_shuffle", b / 1000, seed=s)
        elif exp == "window_shuffle":
            for s in seeds:
                yield exp, "window_shuffle", np.nan, s, SpikeTimingPerturbation(
                    "window_shuffle", seed=s)
        else:
            raise ValueError(exp)


def key(exp, pert, b, seed, session):
    return (exp, pert, "nan" if pd.isna(b) else f"{float(b):g}", int(seed), session)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="results/<out>.csv")
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--experiments", nargs="+",
                    default=["baseline", "snap", "bin_shuffle", "window_shuffle"])
    ap.add_argument("--bins-ms", nargs="+", type=float, default=ALL_BINS_MS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--root", default=str(DATA_ROOT))
    args = ap.parse_args()

    sessions = args.sessions or sorted(
        s for s in TRAIN_RECORDING_IDS if (Path(args.root) / BRAINSET / f"{s}.h5").exists())
    out = PROJECT / "results" / f"{args.out}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    old = (pd.read_csv(out, float_precision="round_trip") if out.exists()
           else pd.DataFrame(columns=COLUMNS))
    done = {key(r.experiment, r.perturbation, r.bin_ms, r.seed, r.session)
            for r in old.itertuples()}
    rows = old.to_dict("records")
    print(f"{len(sessions)} sessions; resuming with {len(rows)} existing rows", flush=True)

    torch.manual_seed(0)
    np.random.seed(0)
    model = load_model(DEFAULT_CKPT, "cuda")
    wrapped, nested = build_dataset(model, args.root, sessions)
    t_start = time.perf_counter()

    def save():
        pd.DataFrame(rows, columns=COLUMNS).to_csv(out, index=False)

    # 1) baseline: always run (paired bootstrap needs its predictions)
    boot = {}
    stored = old[old.experiment == "baseline"].set_index("session").r2
    for s, r in iter_sessions(model, wrapped, nested, sessions, None):
        boot[s] = PairedTrialBootstrap(s, r["timestamps"], r["target"], r["interval"],
                                       r["pred"], n_boot=args.n_boot)
        if s in stored.index:
            assert stored[s] == r["r2"], f"{s}: baseline not reproduced on resume"
        k = key("baseline", "none", 0.0, -1, s)
        if k not in done:
            rows.append(dict(experiment="baseline", session=s, task=TASK(s),
                             perturbation="none", bin_ms=0.0, seed=-1, r2=r["r2"],
                             n_windows=r["n_windows"], n_scored=r["n_scored"],
                             runtime_s=round(r["runtime_s"], 3), dr2=0.0, dr2_boot_se=0.0,
                             dr2_ci_lo=0.0, dr2_ci_hi=0.0))
            done.add(k)
    save()
    print(f"baseline done ({time.perf_counter() - t_start:.0f} s)", flush=True)

    # 2) perturbations, checkpointing after every (setting, session)
    for exp, pert, b, seed, tf in settings(args.experiments, args.bins_ms, args.seeds):
        todo = [s for s in sessions if key(exp, pert, b, seed, s) not in done]
        if not todo:
            continue
        for s, r in iter_sessions(model, wrapped, nested, todo, tf):
            d, se, lo, hi = boot[s].delta(r["timestamps"], r["pred"])
            rows.append(dict(experiment=exp, session=s, task=TASK(s), perturbation=pert,
                             bin_ms=b, seed=seed, r2=r["r2"], n_windows=r["n_windows"],
                             n_scored=r["n_scored"], runtime_s=round(r["runtime_s"], 3),
                             dr2=d, dr2_boot_se=se, dr2_ci_lo=lo, dr2_ci_hi=hi))
            done.add(key(exp, pert, b, seed, s))
            save()
        print(f"{pert:14s} b={b!s:>6} seed={seed:2d}: {len(todo)} sessions "
              f"({time.perf_counter() - t_start:.0f} s elapsed)", flush=True)

    print(f"saved {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
