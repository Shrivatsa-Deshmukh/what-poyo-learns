"""Unit embeddings: statistics of the embedding space and controls for the probes (CPU only).

  count control      does the rate probe only reflect how often a unit fired in training?
                     R² of log rate from log training spike count alone, and with the probe's
                     held-out prediction added (results/emb_probe_preds.csv.gz); partial r
  variance shares    in-sample share of embedding variance explained by session, monkey, area and
                     seven unit properties
  same electrode     cosine similarity of units on the same electrode in two sessions minus that of
                     units on different electrodes, per monkey, with a null that shuffles channel
                     labels within sessions (100 times); and per session pair against days apart

Outputs: results/emb_stats.json, results/emb_same_electrode_pairs.csv (per session pair, for
Figure S2D). Run after scripts/embeddings/run_probes.py.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_probes  # noqa: E402
from src import embedding_probes as ep  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]


def count_control(d):
    p = pd.read_csv(PROJECT / "results/emb_probe_preds.csv.gz")
    p = p[p.target == "log10_rate"]
    cnt = np.log10(d.set_index("unit_id").loc[p.unit_id, "spike_count_whole"].values + 1.0)
    y, yp = p.y.values, p.pred.values
    sst = ((y - y.mean()) ** 2).sum()

    def resid(v, cols):
        A = np.column_stack([np.ones(len(v))] + cols)
        return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    return dict(r2_probe=ep.r2(y, yp),
                r2_count_only=float(1 - (resid(y, [cnt]) ** 2).sum() / sst),
                r2_probe_plus_count=float(1 - (resid(y, [cnt, yp]) ** 2).sum() / sst),
                partial_r=float(np.corrcoef(resid(y, [cnt]), resid(yp, [cnt]))[0, 1]))


def variance_shares(d, X):
    """Session, monkey and area shares; and a linear model of seven unit properties (log rate,
    ISI CV, rank tuning strength, depth / rate, cos / sin preferred direction for well-tuned units,
    well-tuned flag), alone and with session."""
    tr = d.tuning_r2_100.values
    wt = d.well_tuned.astype(bool).values
    P = np.column_stack([d.log10_rate, d.isi_cv,
                         ep.ecdf_normal_scores(tr[np.isfinite(tr)], tr), d.tuning_depth_norm,
                         np.where(wt, np.cos(d.pref_dir), 0), np.where(wt, np.sin(d.pref_dir), 0),
                         wt.astype(float)])
    P = np.where(np.isfinite(P), P, np.nanmean(P, axis=0))
    tot = ((X - X.mean(0)) ** 2).sum()

    def r2_of(A):
        A = np.column_stack([np.ones(len(A)), A])
        return float(1 - ((X - A @ np.linalg.lstsq(A, X, rcond=None)[0]) ** 2).sum() / tot)
    dummies = pd.get_dummies(d.session.values).values[:, 1:].astype(float)
    return dict(session=ep.variance_explained(X, d.session.values)[1],
                monkey=ep.variance_explained(X, d.monkey.values)[1],
                area=ep.variance_explained(X, d.area.values)[1],
                properties=r2_of(P), properties_plus_session=r2_of(np.column_stack([P, dummies])))


def same_electrode(d, X, n_shuffle=100):
    """Mean over session pairs of a monkey of (same-channel minus different-channel cosine
    similarity), with a null that shuffles channel labels among the units of each session."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    rng = np.random.default_rng(20261003)
    res = {}
    for mk, g in d.groupby("monkey"):
        sess, ch = g.session.to_numpy(dtype=str), g.channel.to_numpy(dtype=str)
        Y = Xn[g.index.values]
        us = np.unique(sess)
        idx = {s: np.flatnonzero(sess == s) for s in us}
        blocks = [(idx[a], idx[b], Y[idx[a]] @ Y[idx[b]].T)
                  for i, a in enumerate(us) for b in us[i + 1:]]

        def stat(chan):
            diffs = []
            for i1, i2, S in blocks:
                same = chan[i1][:, None] == chan[i2][None, :]
                if 0 < same.sum() < same.size:
                    diffs.append(S[same].mean() - S[~same].mean())
            return float(np.mean(diffs)), len(diffs)

        obs, n_pairs = stat(ch)
        null = []
        for _ in range(n_shuffle):
            chs = ch.copy()
            for s in us:
                chs[idx[s]] = rng.permutation(chs[idx[s]])
            null.append(stat(chs)[0])
        res[mk] = dict(mean_diff=obs, n_session_pairs=n_pairs, null_mean=float(np.mean(null)),
                       p=float((1 + (np.array(null) >= obs).sum()) / (1 + n_shuffle)))
    return res


def same_electrode_pairs(d, X):
    """Per session pair: same- minus different-electrode similarity and days apart (observed and
    with channel labels shuffled once within sessions)."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    rng = np.random.default_rng(1)
    rows = []
    for mk, g in d.groupby("monkey"):
        sess = g.session.to_numpy(dtype=str)
        ch = g.channel.to_numpy(dtype=str)
        Y = Xn[g.index.values]
        us = np.unique(sess)
        idx = {s: np.flatnonzero(sess == s) for s in us}
        chs = ch.copy()
        for s in us:
            chs[idx[s]] = rng.permutation(chs[idx[s]])
        for i, s1 in enumerate(us):
            for s2 in us[i + 1:]:
                S = Y[idx[s1]] @ Y[idx[s2]].T
                for lab, chan in (("observed", ch), ("shuffled", chs)):
                    same = chan[idx[s1]][:, None] == chan[idx[s2]][None, :]
                    if 0 < same.sum() < same.size:
                        gap = abs((pd.Timestamp(s1[2:10]) - pd.Timestamp(s2[2:10])).days)
                        rows.append(dict(monkey=mk, kind=lab, gap_days=gap,
                                         diff=S[same].mean() - S[~same].mean()))
    df = pd.DataFrame(rows)
    df.to_csv(PROJECT / "results/emb_same_electrode_pairs.csv", index=False)
    obs = df[df.kind == "observed"]
    return {mk: float(g[["diff", "gap_days"]].corr(method="spearman").iloc[0, 1])
            for mk, g in obs.groupby("monkey")}


def main():
    d, X = run_probes.load()
    out = dict(count_control=count_control(d), variance_shares=variance_shares(d, X),
               same_electrode=same_electrode(d, X),
               same_electrode_spearman_vs_days=same_electrode_pairs(d, X))
    (PROJECT / "results/emb_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
