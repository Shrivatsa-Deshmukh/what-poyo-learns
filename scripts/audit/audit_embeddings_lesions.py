"""Independent cross-checks of the embedding study and the lesion study.

Unit properties: spike counts and rates for a few units recomputed with the library's
isin_interval. Probes: the electrode-grouped log-rate probe re-implemented in plain scikit-learn;
within-session R² and the session share of embedding variance recomputed; the same-electrode
statistic recomputed with explicit loops. Lesions: run integrity, the remove-vs-silence summary
and the ridge relative losses recomputed from raw rows.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts" / "embeddings"))
sys.path.insert(0, str(PROJECT / "scripts" / "lesions"))

import run_probes  # noqa: E402
from src import unit_properties as up  # noqa: E402
from src.evaluate import DATA_ROOT, SubsetPoyoMPDataset  # noqa: E402
from torch_brain.data import Interval  # noqa: E402
from torch_brain.utils import isin_interval  # noqa: E402

R = PROJECT / "results"
ORDER = ["frac10", "frac25", "frac50", "frac75", "keep30", "keep20"]
ALPHAS = np.logspace(-2, 4, 13)


def check_unit_rates():
    print("=== unit-property rate spot-check (library isin_interval on the moving scope)")
    props = pd.read_csv(R / "unit_properties.csv")
    s = "c_20131009_random_target_reaching"
    data = SubsetPoyoMPDataset(str(DATA_ROOT), [s]).get_recording(s)
    _, vel, edges = up.bin_recording(data)
    mov, thr = up.moving_intervals(edges, vel, up.used_intervals(data, "train", "scored"))
    train = Interval(start=np.asarray(data.train_domain.start),
                     end=np.asarray(data.train_domain.end))
    iv = train & mov
    st, su = np.asarray(data.spikes.timestamps), np.asarray(data.spikes.unit_index)
    dur = float(np.sum(iv.end - iv.start))
    worst = 0
    for u in (0, 5, 17, 40, 73):
        n_in = int(isin_interval(st[su == u], iv).sum())  # O(n m) memory, fine for one unit
        stored = props[(props.session == s) & (props.local_index == u)].iloc[0]
        worst = max(worst, abs(n_in / dur - stored.rate_hz))
        assert n_in == stored.spike_count, (u, n_in, stored.spike_count)
    print(f"  spike counts identical for 5 units; max |rate diff| = {worst:.2e}; "
          f"threshold {thr:.2f} == stored {stored.speed_threshold:.2f}")


def normal_scores(train_y):
    """Map values to normal scores through the empirical CDF of train_y (own implementation)."""
    srt = np.sort(train_y)
    n = len(srt)
    q = (np.arange(1, n + 1) - 0.5) / n
    return lambda v: ndtri(np.clip(np.interp(v, srt, q), 0.5 / n, 1 - 0.5 / n))


def check_probes(d, X):
    print("=== electrode-grouped rate probe, independent implementation")
    y = d.log10_rate.values
    ok = np.isfinite(y)
    elec = (d.monkey + "/" + d.channel).values
    strat = (d.monkey + "_" + d.task).values
    folds = np.full(len(d), -1)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    for k, (_, te) in enumerate(cv.split(d, strat, groups=elec)):
        folds[te] = k
    pred, yt = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for k in range(5):
        trm, tem = (folds != k) & ok, (folds == k) & ok
        to_ns = normal_scores(y[trm])
        ytr = to_ns(y[trm])  # targets transformed with the whole training fold's ECDF
        # independent penalty choice: inner 3-fold grouped by electrode, minimum MSE
        best, best_a = np.inf, None
        for a in ALPHAS:
            mse = 0
            for itr, ite in GroupKFold(3).split(X[trm], groups=elec[trm]):
                sc = StandardScaler().fit(X[trm][itr])
                m = Ridge(alpha=a).fit(sc.transform(X[trm][itr]), ytr[itr])
                mse += ((m.predict(sc.transform(X[trm][ite])) - ytr[ite]) ** 2).sum()
            if mse < best:
                best, best_a = mse, a
        sc = StandardScaler().fit(X[trm])
        m = Ridge(alpha=best_a).fit(sc.transform(X[trm]), ytr)
        pred[tem] = m.predict(sc.transform(X[tem]))
        yt[tem] = to_ns(y[tem])
    r2 = 1 - np.nansum((yt - pred) ** 2) / np.nansum((yt - np.nanmean(yt)) ** 2)
    stored = pd.read_csv(R / "emb_probe_scores.csv")
    row = stored[(stored.target == "log10_rate") & (stored.subset == "all")].iloc[0]
    print(f"  independent R2 = {r2:.4f} | stored {row.score:.4f} | diff {abs(r2 - row.score):.1e}")

    # within-session R² from the saved predictions
    preds = pd.read_csv(R / "emb_probe_preds.csv.gz")
    g = preds[preds.target == "log10_rate"].copy()
    yd = g.y - g.groupby("session").y.transform("mean")
    pdm = g.pred - g.groupby("session").pred.transform("mean")
    within = 1 - ((yd - pdm) ** 2).sum() / (yd ** 2).sum()
    print(f"  within-session R2 recomputed {within:.4f} | stored {row.within_session_r2:.4f}")

    # session share of embedding variance, plain numpy
    mu = X.mean(0)
    tot = ((X - mu) ** 2).sum()
    gm = pd.DataFrame(X).groupby(d.session.values).transform("mean").values
    share = ((gm - mu) ** 2).sum() / tot
    c = json.load(open(R / "emb_stats.json"))["variance_shares"]["session"]
    print(f"  session variance share recomputed {share:.5f} | stored {c:.5f}")


def check_same_electrode(d, X):
    print("=== same-electrode pairs, monkey M, explicit loops")
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    gM = d[d.monkey == "M"]
    sessions = sorted(gM.session.unique())
    diffs = []
    for i, a in enumerate(sessions):
        for b in sessions[i + 1:]:
            A, B = gM[gM.session == a], gM[gM.session == b]
            S = Xn[A.index] @ Xn[B.index].T
            same = (A.channel.to_numpy(dtype=str)[:, None]
                    == B.channel.to_numpy(dtype=str)[None, :])
            if 0 < same.sum() < same.size:
                diffs.append(S[same].mean() - S[~same].mean())
    stored = json.load(open(R / "emb_stats.json"))["same_electrode"]["M"]["mean_diff"]
    print(f"  recomputed {np.mean(diffs):.5f} over {len(diffs)} session pairs | "
          f"stored {stored:.5f}")


def check_lesion_summary():
    print("=== lesions: remove vs silence, recomputed")
    runs = pd.read_csv(R / "lesion.csv", float_precision="round_trip")
    sw = pd.read_csv(R / "sweep.csv", float_precision="round_trip")
    base = sw[sw.experiment == "baseline"].set_index("session").r2
    assert np.allclose(runs.dr2, runs.r2 - runs.session.map(base), atol=1e-6), \
        "dr2 != r2 - baseline"
    idx = ["session", "condition", "seed"]
    nl = runs.pivot_table(index=idx, columns="lesion_type", values="n_lesioned")
    assert (nl["remove"] == nl["silence"]).all(), "different lesion sizes for remove and silence"
    wide = runs.pivot_table(index=idx, columns="lesion_type", values="dr2").reset_index()
    wide["diff"] = wide["remove"] - wide["silence"]
    ses = wide.groupby(["session", "condition"])["diff"].mean().reset_index()
    p = ses.groupby("condition")["diff"].apply(lambda x: wilcoxon(x).pvalue).reindex(ORDER)
    holm = multipletests(p.values, method="holm")[1]
    summ = pd.read_csv(R / "lesion_summary.csv")
    summ = summ[summ.task == "all"].set_index("condition").reindex(ORDER)
    mean_diff = ses.groupby("condition")["diff"].mean().reindex(ORDER).values
    print("  max |diff_mean - stored| =", f"{np.abs(mean_diff - summ.diff_mean.values).max():.1e}",
          "| max |holm - stored| =", f"{np.abs(holm - summ.wilcoxon_p_holm.values).max():.1e}")


def check_ridge_loss():
    rr = pd.read_csv(R / "lesion_ridge.csv")
    base = rr[rr.condition == "none"].set_index(["session", "variant"]).r2_test
    rr = rr[(rr.condition != "none") & (rr.variant == "grid")].copy()
    rr["rl"] = [(base[(s, "grid")] - v) / base[(s, "grid")] for s, v in zip(rr.session, rr.r2_test)]
    rel = rr.groupby(["condition", "session"]).rl.mean().groupby("condition").mean().reindex(ORDER)
    vr = pd.read_csv(R / "lesion_vs_ridge.csv")
    vr = vr[vr.variant == "grid"].set_index("condition").reindex(ORDER)
    print(f"  ridge relative loss max |diff| = "
          f"{np.abs(rel.values - vr.ridge_refit_rel_loss.values).max():.1e}")


if __name__ == "__main__":
    check_unit_rates()
    d, X = run_probes.load()
    check_probes(d, X)
    check_same_electrode(d, X)
    check_lesion_summary()
    check_ridge_loss()
