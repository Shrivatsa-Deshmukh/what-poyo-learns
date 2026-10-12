"""Lesion study, CPU part: what did POYO-MP see during training under UnitDropout?

Training transform (checkpoint hyper_parameters.train_transforms):
  UnitDropout(max_units=300, min_units=30, mode_units=100, peak=4)
which, per training window, draws k = int(TriangleDistribution.sample(n)) and keeps the first k
of a random permutation of the session's n units (all n if k ≥ n), removing the others from
data.units together with their spikes (torch_brain/transforms/unit_dropout.py:130-162).
Here the library's own TriangleDistribution is sampled N_DRAWS times per session.

Per session: kept-count distribution (quantiles), P(a window drops ≥ 1 unit), P(a given unit is
dropped in a window) = 1 − E[kept] / n. Per lesion condition (fractions 10/25/50/75% and kept
counts 30/20 where < n): kept count, its training CDF P_train(kept ≤ k), and
in_training = P_train(kept ≤ k) ≥ 1% (the model saw at least 1% of windows with this few units).

Outputs: results/lesion_dropout_sim.csv (per session), results/lesion_conditions.csv (per session ×
condition)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torch_brain.transforms import TriangleDistribution  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
N_DRAWS = 20000
FRACTIONS = [0.10, 0.25, 0.50, 0.75]
KEEP_COUNTS = [30, 20]
IN_TRAINING_MIN_CDF = 0.01


def kept_counts(n, rng_seed):
    dist = TriangleDistribution(min_units=30, mode_units=100, max_units=300, peak=4,
                                seed=rng_seed)
    k = np.array([int(dist.sample(n)) for _ in range(N_DRAWS)])
    return np.minimum(k, n)  # permutation(n)[:k] keeps all n units when k >= n


def conditions(n):
    out = [(f"frac{int(f * 100)}", "fraction", f, n - int(round(f * n))) for f in FRACTIONS]
    out += [(f"keep{k}", "keep_count", np.nan, k) for k in KEEP_COUNTS if k < n]
    return out


def main():
    units = pd.read_csv(PROJECT / "results/emb_units.csv")
    sizes = units.groupby("session").size()
    rows, crow = [], []
    for i, (s, n) in enumerate(sizes.items()):
        k = kept_counts(int(n), rng_seed=1000 + i)
        rows.append(dict(session=s, monkey=s[0].upper(), task="CO" if "center_out" in s else "RT",
                         n_units=int(n), p_window_drops_any=float((k < n).mean()),
                         p_unit_dropped=float(1 - k.mean() / n), kept_mean=float(k.mean()),
                         kept_q01=float(np.percentile(k, 1)), kept_q50=float(np.median(k)),
                         kept_q99=float(np.percentile(k, 99)), kept_min=int(k.min())))
        for name, kind, f, kept in conditions(int(n)):
            cdf = float((k <= kept).mean())
            crow.append(dict(session=s, n_units=int(n), condition=name, kind=kind, fraction=f,
                             kept=int(kept), n_lesioned=int(n - kept), train_cdf=cdf,
                             in_training=cdf >= IN_TRAINING_MIN_CDF))
    sim = pd.DataFrame(rows)
    cond = pd.DataFrame(crow)
    sim.to_csv(PROJECT / "results/lesion_dropout_sim.csv", index=False)
    cond.to_csv(PROJECT / "results/lesion_conditions.csv", index=False)

    pd.set_option("display.width", 200)
    print("=== training dropout per session (quantiles over sessions)")
    print(sim[["n_units", "p_window_drops_any", "p_unit_dropped", "kept_q01", "kept_q50", "kept_q99"]]
          .describe(percentiles=[.1, .5, .9]).round(3).to_string())
    print("\nsessions with n_units < 30 (training never dropped units):", int((sim.n_units < 30).sum()))
    print("\n=== share of sessions where each lesion condition is inside the training distribution"
          f" (P_train(kept <= k) >= {IN_TRAINING_MIN_CDF:.0%})")
    print(cond.groupby("condition").agg(sessions=("session", "size"),
                                        in_training=("in_training", "mean"),
                                        median_train_cdf=("train_cdf", "median"))
          .reindex([f"frac{int(f * 100)}" for f in FRACTIONS] + [f"keep{k}" for k in KEEP_COUNTS])
          .round(3).to_string())


if __name__ == "__main__":
    main()
