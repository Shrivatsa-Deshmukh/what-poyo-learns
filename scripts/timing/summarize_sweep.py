"""Summarize a scripts/timing/run_perturb.py results CSV.

Per (experiment, bin_ms, task):
  - across-session spread: mean ± SEM of session R² and of session ΔR² (seeds averaged
    within session first), n sessions;
  - seed spread: mean over sessions of the per-session SD across seeds (bin_shuffle, window_shuffle);
  - Wilcoxon signed-rank test (two-sided) of per-session ΔR² against 0, plus Holm
    correction across settings within each task;
  - n sessions (or session×seed rows) whose paired-bootstrap 95% CI excludes 0.

Usage: python scripts/timing/summarize_sweep.py <name>   -> results/summary_<name>.csv (table printed)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ["snap", "bin_shuffle", "window_shuffle"]  # row order of the summary


def holm(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][idx])
    tmp = np.empty(m)
    tmp[idx] = np.minimum(adj, 1.0)
    out[ok] = tmp
    return out


def summarize(df):
    df = df.copy()
    df["bin_ms"] = df.bin_ms.fillna(-1)  # window_shuffle has no bin
    df["ci_excl0"] = (df.dr2_ci_lo > 0) | (df.dr2_ci_hi < 0)
    pert = df[df.experiment != "baseline"]
    sess = (pert.groupby(["experiment", "bin_ms", "task", "session"])
            .agg(r2=("r2", "mean"), dr2=("dr2", "mean"), seed_sd=("r2", "std"),
                 n_seeds=("r2", "size"), n_ci_excl0=("ci_excl0", "sum")).reset_index())
    out = []
    for (exp, b, task), g in sess.groupby(["experiment", "bin_ms", "task"]):
        n = len(g)
        p = wilcoxon(g.dr2).pvalue if n >= 2 and np.any(g.dr2 != 0) else np.nan
        out.append(dict(
            experiment=exp, bin_ms=None if b == -1 else b, task=task, n_sessions=n,
            r2_mean=g.r2.mean(), r2_sem=g.r2.sem() if n > 1 else np.nan,
            dr2_mean=g.dr2.mean(), dr2_sem=g.dr2.sem() if n > 1 else np.nan,
            dr2_median=g.dr2.median(),
            seed_sd_mean=g.seed_sd.mean() if (g.n_seeds > 1).all() else np.nan,
            rows_ci_excl0=int(g.n_ci_excl0.sum()), rows_total=int(g.n_seeds.sum()),
            wilcoxon_p=p))
    res = pd.DataFrame(out)
    order = {e: i for i, e in enumerate(EXPERIMENTS)}
    res = res.sort_values("experiment", key=lambda s: s.map(order), kind="stable",
                          ignore_index=True)
    res["wilcoxon_p_holm"] = np.nan
    for task, g in res.groupby("task"):
        res.loc[g.index, "wilcoxon_p_holm"] = holm(g.wilcoxon_p)
    base = df[df.experiment == "baseline"].groupby("task").r2.agg(["mean", "sem", "size"])
    return res, base


def main():
    name = sys.argv[1]
    df = pd.read_csv(PROJECT / "results" / f"{name}.csv")
    res, base = summarize(df)
    res.to_csv(PROJECT / "results" / f"summary_{name}.csv", index=False)
    lines = ["| experiment | bin (ms) | task | n | R² mean ± SEM | ΔR² mean ± SEM | "
             "seed SD | CI excl. 0 | Wilcoxon p (Holm) |", "|" + "---|" * 9]
    for r in res.itertuples():
        b = "–" if pd.isna(r.bin_ms) else f"{r.bin_ms:g}"
        lines.append(
            f"| {r.experiment} | {b} | {r.task} | {r.n_sessions} | {r.r2_mean:.4f} ± "
            f"{r.r2_sem:.4f} | {r.dr2_mean:+.4f} ± {r.dr2_sem:.4f} | {r.seed_sd_mean:.4f} | "
            f"{r.rows_ci_excl0}/{r.rows_total} | {r.wilcoxon_p:.2g} ({r.wilcoxon_p_holm:.2g}) |")
    md = "Baseline: " + ", ".join(f"{t} {row['mean']:.4f} ± {row['sem']:.4f} (n={int(row['size'])})"
                                  for t, row in base.iterrows()) + "\n\n" + "\n".join(lines)
    print(md)


if __name__ == "__main__":
    main()
