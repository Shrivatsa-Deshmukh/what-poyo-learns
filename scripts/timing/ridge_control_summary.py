"""Summarize the binned ridge control and compare its bin-size curve with POYO's.

Each decoder is compared with its own finest-resolution reference:
  POYO snap / bin shuffle: ΔR² vs the unperturbed POYO baseline (results/sweep.csv)
  ridge:      ΔR² vs the same session's 10 ms ridge of the same variant
Also fractional loss = −ΔR² / R²_ref, and the bin size at which the mean fractional loss
first reaches 1% (log-interpolated).

Output: results/binned_ridge_summary.csv (plotted in scripts/docs/paper_figures.py)
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
BINS = [10, 20, 50, 100, 150, 200, 333, 500]


def per_session_curves():
    rid = pd.read_csv(PROJECT / "results/binned_ridge.csv", float_precision="round_trip")
    ref = rid[rid.bin_ms == 10].set_index(["session", "variant"]).r2_test
    rid["r2_ref"] = [ref[(s, v)] for s, v in zip(rid.session, rid.variant)]
    rid["dr2"] = rid.r2_test - rid.r2_ref
    rid["decoder"] = "ridge (" + rid.variant + ")"
    rid = rid.rename(columns={"r2_test": "r2"})

    sw = pd.read_csv(PROJECT / "results/sweep.csv", float_precision="round_trip")
    base = sw[sw.experiment == "baseline"].set_index("session").r2
    sw = sw[sw.experiment.isin(["snap", "bin_shuffle"]) & sw.bin_ms.isin(BINS)]
    sw = sw.groupby(["experiment", "session", "task", "bin_ms"], as_index=False)[["r2", "dr2"]].mean()
    sw["r2_ref"] = sw.session.map(base)
    sw["decoder"] = sw.experiment.map({"snap": "POYO snap", "bin_shuffle": "POYO bin shuffle"})
    cols = ["decoder", "session", "task", "bin_ms", "r2", "r2_ref", "dr2"]
    d = pd.concat([sw[cols], rid[cols]], ignore_index=True)
    d["frac_loss"] = -d.dr2 / d.r2_ref
    return d


def break_bin(g, thr=0.01):
    """Bin (ms) where mean fractional loss first reaches thr, log-interpolated."""
    g = g.sort_values("bin_ms")
    x, y = np.log(g.bin_ms.values), g.frac_loss.values
    for i in range(1, len(y)):
        if y[i] >= thr > y[i - 1]:
            return float(np.exp(np.interp(thr, [y[i - 1], y[i]], [x[i - 1], x[i]])))
    return np.nan


def main():
    d = per_session_curves()
    s = (d.groupby(["decoder", "task", "bin_ms"])
         .agg(n=("session", "nunique"), r2=("r2", "mean"), r2_sem=("r2", "sem"),
              dr2=("dr2", "mean"), dr2_sem=("dr2", "sem"), frac_loss=("frac_loss", "mean"),
              frac_loss_sem=("frac_loss", "sem"))
         .reset_index())
    s.to_csv(PROJECT / "results/binned_ridge_summary.csv", index=False)
    print(s.round(4).to_string(index=False))
    print("\nbin where mean fractional loss reaches 1% (ms):")
    for (dec, task), g in s.groupby(["decoder", "task"]):
        print(f"  {dec:16s} {task}: {break_bin(g):6.1f}")



if __name__ == "__main__":
    main()
