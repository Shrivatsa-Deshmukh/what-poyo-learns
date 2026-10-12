"""Independent cross-checks of the cross-day transfer analysis.

1. Key statistics recomputed from the raw rows with plain numpy / scipy: median kept shares, the
   matched vs within-area-shuffle test.
2. Model re-runs on the CPU for two sessions, with the swapped vectors built independently of
   scripts/transfer/run_transfer.py (pandas group means; session rows from the vocabulary):
   own vectors, matched and the other-monkey session swap; R² compared with the GPU
   rows of results/transfer.csv.
3. One cross-day ridge redone with scikit-learn (electrode counts, same penalty) and compared
   with results/transfer_ridge.csv.
4. Improved matching (results/transfer_matching.csv, transfer_combined.csv,
   transfer_ridge_pooled.csv): medians and one paired test recomputed with numpy / scipy; donor
   pools rebuilt from the session dates; the same two sessions re-run on the CPU with pooled and
   rate-matched vectors built independently (scipy cdist); one pooled ridge redone with
   scikit-learn.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts" / "transfer"))

import ridge_cross_day as rc  # noqa: E402
from src.binned_decoder import ALPHA_GRID, WeightedRidge  # noqa: E402
from src.bootstrap import r2_np  # noqa: E402
from src.evaluate import (  # noqa: E402
    BRAINSET, DATA_ROOT, DEFAULT_CKPT, build_dataset, iter_sessions, load_model,
)
from src.transfer import swapped_rows  # noqa: E402

R = PROJECT / "results"
SESSIONS = ["m_20140203_center_out_reaching", "c_20131009_random_target_reaching"]


def check_statistics():
    print("=== (1) cross-day statistics recomputed from raw rows")
    d = pd.read_csv(R / "transfer.csv", float_precision="round_trip")
    sw = pd.read_csv(R / "sweep.csv", float_precision="round_trip")
    base = sw[sw.experiment == "baseline"].set_index("session").r2
    d["kept"] = d.r2 / d.session.map(base)
    per = d.groupby(["session", "condition"]).agg(r2=("r2", "mean"), kept=("kept", "mean"))
    summ = pd.read_csv(R / "transfer_summary.csv").query("scope == 'all'").set_index("condition")
    for c in ("matched", "shuffled_within_area", "own_electrode_mean", "session_other_monkey"):
        mine = np.median(per.xs(c, level="condition").kept)
        print(f"  {c:22s} median kept {mine:.6f} | stored {summ.loc[c, 'kept_median']:.6f}")
        assert abs(mine - summ.loc[c, "kept_median"]) < 1e-12
    m = per.xs("matched", level="condition").r2
    s = per.xs("shuffled_within_area", level="condition").r2
    p = wilcoxon((m - s).to_numpy()).pvalue
    t = pd.read_csv(R / "transfer_tests.csv").set_index("contrast")
    stored = t.loc["matched vs shuffled within area (R²)", "wilcoxon_p"]
    print(f"  matched vs within-area shuffle: p {p:.3e} | stored {stored:.3e}")
    assert np.isclose(p, stored, rtol=1e-9)


def check_model_reruns():
    print("=== (2) embedding swaps re-run on the CPU, vectors built independently")
    torch.set_num_threads(16)
    units = pd.read_csv(R / "emb_units.csv")
    runs = pd.read_csv(R / "transfer.csv", float_precision="round_trip")
    model = load_model(DEFAULT_CKPT, "cpu")
    W = model.unit_emb.weight.detach().numpy().astype(np.float64)
    Ws = model.session_emb.weight.detach().numpy().astype(np.float64)
    wrapped, nested = build_dataset(model, str(DATA_ROOT), SESSIONS)
    worst = 0.0
    for s in SESSIONS:
        tgt = units[units.session == s]
        rows = [model.unit_emb.vocab[u] for u in tgt.unit_id]
        srow = model.session_emb.vocab[f"{BRAINSET}/{s}"]
        main = runs[(runs.session == s) & (runs.condition == "matched")].iloc[0]
        dn = units[units.session == main.donor]
        means = dn.groupby("channel").emb_row.apply(lambda r: W[list(r)].mean(0))
        fallback = W[dn.emb_row].mean(0)
        matched = {r: (means[c] if c in means.index else fallback)
                   for r, c in zip(rows, tgt.channel)}
        om = runs[(runs.session == s) & (runs.condition == "session_other_monkey")].iloc[0]
        orow = model.session_emb.vocab[f"{BRAINSET}/{om.donor}"]
        cases = [("own_swap", {r: W[r] for r in rows}, {srow: Ws[srow]}),
                 ("matched", matched, None),
                 ("session_other_monkey", None, {srow: Ws[orow]})]
        for cond, ur, sr in cases:
            with swapped_rows(model, ur, sr):
                _, r = next(iter_sessions(model, wrapped, nested, [s], None, device="cpu"))
            gpu = runs[(runs.session == s) & (runs.condition == cond)].r2.iloc[0]
            worst = max(worst, abs(gpu - r["r2"]))
            print(f"  {s[:24]:24s} {cond:22s} GPU {gpu:.6f} CPU {r['r2']:.6f}")
    print(f"  max |CPU - GPU| = {worst:.1e}")
    assert worst < 1e-6


def sk_predict(donor, s):
    """Donor-day ridge (scikit-learn) applied to session s's test set: (prediction, target)."""
    ts_d, ch_d, sets_d = rc.load(donor)
    channels = sorted(rc.UNITS[rc.UNITS.session == donor].channel.unique())
    Xtr = rc.features(ts_d, ch_d, channels, sets_d["train"][0])
    Xva = rc.features(ts_d, ch_d, channels, sets_d["valid"][0])
    w, y = sets_d["train"][2], sets_d["train"][1]
    mu = (w[:, None] * Xtr).sum(0) / w.sum()
    sd = np.sqrt((w[:, None] * (Xtr - mu) ** 2).sum(0) / w.sum())
    sd[sd < 1e-6] = 1
    scale = WeightedRidge().fit(sp.csr_matrix(Xtr), y, w).scale
    best = max(range(len(ALPHA_GRID)), key=lambda i: r2_np(
        Ridge(alpha=ALPHA_GRID[i] * scale).fit((Xtr - mu) / sd, y, sample_weight=w)
        .predict((Xva - mu) / sd), sets_d["valid"][1]))
    sk = Ridge(alpha=ALPHA_GRID[best] * scale).fit((Xtr - mu) / sd, y, sample_weight=w)
    ts_t, ch_t, sets_t = rc.load(s)
    Xte = rc.features(ts_t, ch_t, channels, sets_t["test"][0])
    present = set(rc.UNITS[rc.UNITS.session == s].channel)
    for i, c in enumerate(channels):
        if c not in present:
            Xte[:, [k * len(channels) + i for k in range(rc.K)]] = mu[
                [k * len(channels) + i for k in range(rc.K)]]
    return sk.predict((Xte - mu) / sd), sets_t["test"][1]


def check_ridge():
    print("=== (3) one cross-day ridge redone with scikit-learn")
    rid = pd.read_csv(R / "transfer_ridge.csv", float_precision="round_trip")
    s = SESSIONS[0]
    row = rid[(rid.session == s) & (rid.condition == "cross_main")].iloc[0]
    donor = row.donor
    r2 = r2_np(*sk_predict(donor, s))
    print(f"  {s[:24]} <- {donor[:24]}: stored R2 {row.r2:.6f} | sklearn {r2:.6f} | "
          f"diff {abs(r2 - row.r2):.1e}")
    assert abs(r2 - row.r2) < 1e-6


def date(s):
    return pd.Timestamp(s[2:10])


def pool_of(s, units, k=3):
    """The k nearest same-monkey, same-task days (ties by name)."""
    meta = units.drop_duplicates("session").set_index("session")
    cand = [x for x in meta.index if x != s and x[0] == s[0] and meta.loc[x, "task"] == meta.loc[s, "task"]]
    return sorted(cand, key=lambda x: (abs((date(x) - date(s)).days), x))[:k]


def check_improved_statistics():
    print("=== (4a) improved matching: statistics and pools")
    sw = pd.read_csv(R / "sweep.csv", float_precision="round_trip")
    base = sw[sw.experiment == "baseline"].set_index("session").r2
    m = pd.read_csv(R / "transfer_matching.csv", float_precision="round_trip")
    c = pd.read_csv(R / "transfer_combined.csv", float_precision="round_trip")
    x = pd.concat([m, c])
    x["kept"] = x.r2 / x.session.map(base)
    summ = pd.read_csv(R / "transfer_improved_summary.csv").query("scope == 'all'").set_index("condition")
    for cond in ("matched", "matched_3days", "electrode_rate", "matched_3days_rate", "rate_only"):
        mine = np.median(x[x.condition == cond].kept.to_numpy())
        print(f"  {cond:20s} median kept {mine:.6f} | stored {summ.loc[cond, 'kept_median']:.6f}")
        assert abs(mine - summ.loc[cond, "kept_median"]) < 1e-12
    k = x[x.gap.isna()].pivot(index="session", columns="condition", values="kept")
    p = wilcoxon((k.matched_3days_rate - k.matched).to_numpy()).pvalue
    t = pd.read_csv(R / "transfer_improved_tests.csv").set_index("contrast")
    stored = t.loc["matched_3days_rate vs matched (share of own R² kept)", "wilcoxon_p"]
    print(f"  3 days + rate vs matched: p {p:.3e} | stored {stored:.3e}")
    assert np.isclose(p, stored, rtol=1e-9)
    rp = pd.read_csv(R / "transfer_ridge_pooled.csv", float_precision="round_trip")
    units = pd.read_csv(R / "emb_units.csv")
    main = rp[rp.condition == "cross_3days"]
    bad = sum(row.pool != "+".join(pool_of(row.session, units)) for row in main.itertuples())
    print(f"  ridge main pools rebuilt from dates: {len(main) - bad}/{len(main)} identical")
    assert bad == 0
    mine = np.median(rp.r2[rp.condition == "cross_3days"] / rp.r2_own_day[rp.condition == "cross_3days"])
    stored = summ.loc["ridge_cross_3days", "kept_median"]
    print(f"  ridge 3 days median kept {mine:.6f} | stored {stored:.6f}")
    assert abs(mine - stored) < 1e-12


def check_improved_reruns():
    print("=== (4b) pooled and rate-matched swaps re-run on the CPU, vectors built independently")
    torch.set_num_threads(16)
    units = pd.read_csv(R / "emb_units.csv")
    props = pd.read_csv(R / "unit_properties.csv", usecols=["unit_id", "log10_rate_whole", "isi_cv_whole"])
    f = units[["unit_id"]].merge(props, on="unit_id", how="left").drop(columns="unit_id")
    f = f.fillna(f.median())
    units[["f1", "f2"]] = ((f - f.mean()) / f.std()).to_numpy()
    rp = pd.read_csv(R / "transfer_ridge_pooled.csv", float_precision="round_trip")
    runs = pd.concat([pd.read_csv(R / n, float_precision="round_trip")
                      for n in ("transfer_matching.csv", "transfer_combined.csv")])
    model = load_model(DEFAULT_CKPT, "cpu")
    W = model.unit_emb.weight.detach().numpy().astype(np.float64)
    wrapped, nested = build_dataset(model, str(DATA_ROOT), SESSIONS)

    def build(tgt, donors, rate):
        dn = units[units.session.isin(donors)]
        fallback = W[dn.emb_row].mean(0)
        out = {}
        for u in tgt.itertuples():
            on = dn[dn.channel == u.channel]
            if not rate:
                out[u.emb_row] = W[on.emb_row].mean(0) if len(on) else fallback
                continue
            if not len(on):
                on = dn[dn.area == u.area] if (dn.area == u.area).any() else dn
            j = cdist([[u.f1, u.f2]], on[["f1", "f2"]].to_numpy(), "sqeuclidean")[0].argmin()
            out[u.emb_row] = W[on.emb_row.iloc[j]]
        return out

    worst = 0.0
    for s in SESSIONS:
        tgt = units[units.session == s]
        main = runs[(runs.session == s) & (runs.condition == "matched")].iloc[0].donor
        gap_row = runs[(runs.session == s) & (runs.condition == "matched_3days_rate_gap")].iloc[0]
        gpool = rp[(rp.session == s) & (rp.gap == gap_row.gap)].iloc[0].pool.split("+")
        assert gpool[0] == gap_row.donor
        cases = [("matched_3days", None, build(tgt, pool_of(s, units), False)),
                 ("electrode_rate", None, build(tgt, [main], True)),
                 ("matched_3days_rate", None, build(tgt, pool_of(s, units), True)),
                 ("matched_3days_rate_gap", gap_row.gap, build(tgt, gpool, True))]
        for cond, gap, ur in cases:
            with swapped_rows(model, ur, None):
                _, r = next(iter_sessions(model, wrapped, nested, [s], None, device="cpu"))
            sel = (runs.session == s) & (runs.condition == cond)
            if gap:
                sel &= runs.gap == gap
            gpu = runs[sel].r2.iloc[0]
            worst = max(worst, abs(gpu - r["r2"]))
            print(f"  {s[:24]:24s} {cond:24s} GPU {gpu:.6f} CPU {r['r2']:.6f}")
    print(f"  max |CPU - GPU| = {worst:.1e}")
    assert worst < 1e-6


def check_pooled_ridge():
    print("=== (4c) one pooled cross-day ridge redone with scikit-learn")
    rp = pd.read_csv(R / "transfer_ridge_pooled.csv", float_precision="round_trip")
    s = SESSIONS[0]
    row = rp[(rp.session == s) & (rp.condition == "cross_3days")].iloc[0]
    preds = [sk_predict(d, s) for d in row.pool.split("+")]
    r2 = r2_np(np.mean([p for p, _ in preds], axis=0), preds[0][1])
    print(f"  {s[:24]} <- {row.n_days} days: stored R2 {row.r2:.6f} | sklearn {r2:.6f} | "
          f"diff {abs(r2 - row.r2):.1e}")
    assert abs(r2 - row.r2) < 1e-6


if __name__ == "__main__":
    check_statistics()
    check_ridge()
    check_improved_statistics()
    check_pooled_ridge()
    check_model_reruns()
    check_improved_reruns()
