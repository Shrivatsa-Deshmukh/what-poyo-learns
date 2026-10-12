"""Independent cross-check of the timing-study results (run from the project root).

1. Sweep summary statistics rebuilt from results/sweep.csv with fresh code and statsmodels Holm.
2. Three ridge fits redone with scikit-learn Ridge (same features, penalty and weights).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.binned_decoder import ALPHA_GRID, WeightedRidge, count_features, session_sets  # noqa: E402
from src.bootstrap import r2_np  # noqa: E402
from src.evaluate import DATA_ROOT, PROJECT, SubsetPoyoMPDataset  # noqa: E402

R = PROJECT / "results"
RIDGE_CASES = [("c_20131003_center_out_reaching", "grid", 50.0),
               ("m_20140114_random_target_reaching", "grid", 150.0),
               ("j_20160405_center_out_reaching", "grid", 333.0)]


def check_sweep_summary():
    print("=== (1) sweep summary recomputed independently")
    sw = pd.read_csv(R / "sweep.csv", float_precision="round_trip")
    base = sw[sw.experiment == "baseline"].set_index("session").r2
    p = sw[sw.experiment != "baseline"].copy()
    p["dr2_direct"] = p.r2 - p.session.map(base)
    print("  max |dr2 (bootstrap, float64) - (r2 - r2_base) (torchmetrics float32)|:",
          f"{(p.dr2 - p.dr2_direct).abs().max():.1e}")
    p["bin"] = p.bin_ms.fillna(-1)
    ses = (p.groupby(["experiment", "bin", "task", "session"])
           .agg(dr2=("dr2", "mean"), r2=("r2", "mean"), sd=("r2", "std")).reset_index())
    mine = []
    for (e, b, t), g in ses.groupby(["experiment", "bin", "task"]):
        mine.append(dict(experiment=e, bin=b, task=t, n=len(g), dr2_mean=g.dr2.mean(),
                         dr2_sem=g.dr2.std(ddof=1) / np.sqrt(len(g)),
                         p=wilcoxon(g.dr2).pvalue, seed_sd=g.sd.mean()))
    mine = pd.DataFrame(mine)
    for _, g in mine.groupby("task"):
        mine.loc[g.index, "p_holm"] = multipletests(g.p, method="holm")[1]
    stored = pd.read_csv(R / "summary_sweep.csv")
    stored["bin"] = stored.bin_ms.fillna(-1)
    m = mine.merge(stored, on=["experiment", "bin", "task"])
    assert len(m) == len(mine) == len(stored)
    for a, b in (("dr2_mean_x", "dr2_mean_y"), ("dr2_sem_x", "dr2_sem_y"), ("p", "wilcoxon_p"),
                 ("p_holm", "wilcoxon_p_holm"), ("seed_sd", "seed_sd_mean")):
        print(f"  {b:16s} max |diff| = {(m[a] - m[b]).abs().max():.2e}")
    print("  n per cell:", sorted(set(m["n"])))


def check_ridge():
    print("=== (2) ridge refit with sklearn (same features, alpha, weights)")
    rid = pd.read_csv(R / "binned_ridge.csv", float_precision="round_trip")
    for session, variant, b in RIDGE_CASES:
        row = rid[(rid.session == session) & (rid.variant == variant) & (rid.bin_ms == b)].iloc[0]
        data = SubsetPoyoMPDataset(str(DATA_ROOT), [session]).get_recording(session)
        sets = session_sets(data)
        ts, un = np.asarray(data.spikes.timestamps), np.asarray(data.spikes.unit_index)
        n_units, K = len(data.units.id), int(row.K)
        Xtr = count_features(ts, un, n_units, sets["train"][0], b / 1000, K, variant).toarray()
        Xte = count_features(ts, un, n_units, sets["test"][0], b / 1000, K, variant).toarray()
        w = sets["train"][2]
        mu = (w[:, None] * Xtr).sum(0) / w.sum()
        sd = np.sqrt((w[:, None] * (Xtr - mu) ** 2).sum(0) / w.sum())
        sd[sd < 1e-6] = 1
        ours = WeightedRidge().fit(sp.csr_matrix(Xtr), sets["train"][1], w)
        lam = ALPHA_GRID[int(row.alpha_idx)] * ours.scale
        sk = Ridge(alpha=lam, fit_intercept=True).fit((Xtr - mu) / sd, sets["train"][1],
                                                      sample_weight=w)
        r2_sk = r2_np(sk.predict((Xte - mu) / sd), sets["test"][1])
        print(f"  {session[:22]} {variant:7s} {b:5.0f} ms: stored R2 {row.r2_test:.6f} | "
              f"sklearn {r2_sk:.6f} | diff {abs(r2_sk - row.r2_test):.1e}")


if __name__ == "__main__":
    check_sweep_summary()
    check_ridge()
