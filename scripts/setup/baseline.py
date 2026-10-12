"""POYO-MP baseline R² on the current (torch_brain 0.2.0) test split, per session.

Sampling intervals come from the library: Dataset.get_sampling_intervals("test")
(= the stored test_domain). Evaluation path: src/evaluate.py (example's validation loop).

Usage: python scripts/setup/baseline.py [--sessions s1 s2 ...]   (default: all POYO-MP sessions
that are prepared on disk)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluate import (  # noqa: E402
    BRAINSET, DATA_ROOT, DEFAULT_CKPT, PROJECT, TRAIN_RECORDING_IDS, build_dataset, evaluate,
    load_model,
)

SEED = 0
PAPER = {"CO": 0.95, "RT": 0.87}  # POYO-mp multi-session averages (user-provided)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--root", default=str(DATA_ROOT))
    ap.add_argument("--out", default=str(PROJECT / "results" / "baseline.csv"))
    args = ap.parse_args()

    sessions = args.sessions or sorted(
        s for s in TRAIN_RECORDING_IDS
        if (Path(args.root) / BRAINSET / f"{s}.h5").exists()
    )
    missing = [s for s in TRAIN_RECORDING_IDS if s not in sessions]
    print(f"{len(sessions)} sessions ({len(missing)} POYO-MP sessions not included)")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = load_model(DEFAULT_CKPT, "cuda")
    wrapped, _ = build_dataset(model, args.root, sessions)
    intervals = wrapped.get_sampling_intervals("test")
    assert len(intervals) == len(sessions)

    res = evaluate(model, wrapped, intervals, device="cuda")
    rows = []
    for rid, r in sorted(res.items()):
        s = rid.split("/", 1)[1]
        rows.append(dict(experiment="baseline", session=s,
                         task="CO" if "center_out" in s else "RT",
                         perturbation="none", bin_ms=0, seed=-1, r2=r["r2"],
                         n_windows=r["n_windows"], n_scored=r["n_scored"]))
    df = pd.DataFrame(rows)
    out = Path(args.out)
    df.to_csv(out, index=False)

    summ = df.groupby("task").r2.agg(["count", "mean", "std", "sem", "min", "median", "max"])
    summ["paper"] = summ.index.map(PAPER)
    print(summ.round(4).to_string())
    print("\nlowest 5 sessions:")
    print(df.nsmallest(5, "r2")[["session", "task", "r2", "n_scored"]].to_string(index=False))

    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
