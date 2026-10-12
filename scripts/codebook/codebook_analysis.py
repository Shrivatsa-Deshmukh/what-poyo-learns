"""Neuron letters analysis (CPU only, from saved results and the checkpoint's embedding table).

Inputs: results/codebook.csv and, if present, results/codebook_direction.csv
(scripts/codebook/run_codebook.py), results/emb_units.csv, checkpoints/poyo_mp.ckpt.
"Kept" = R² of a condition ÷ the session's own R² at the same precision (r2_base, fp16).

Outputs:
  results/codebook_summary.csv   per scheme x condition, all sessions and per monkey: sessions,
                                 median kept [session-bootstrap 95% CI], quartiles, share R² > 0,
                                 median coding error
  results/codebook_tests.csv     paired Wilcoxon tests (per session) of the key contrasts: equal
                                 bits, plateau, letters vs continuous, other sessions vs other
                                 monkeys, direction-only vs whole vector; Holm-corrected
  results/codebook_letters.csv   descriptive letters fitted on ALL units (not held out; for
                                 reading the letters only): adjusted mutual information of the
                                 letter with monkey / session / area / task, and how often units
                                 on the same electrode on different days share a letter
  results/codebook_stats.json    embedding length by monkey; distance of each unit to its nearest
                                 unit of another session of the same monkey vs of another monkey
The report's figure is drawn by scripts/docs/paper_figures.py (fig_letters).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import adjusted_mutual_info_score
from statsmodels.stats.multitest import multipletests

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from src.codebook import fit_centroids, fit_rotation, blocks, nearest, unit_rows  # noqa: E402

R = PROJECT / "results"
SEED = 20261010

TESTS = [  # (scheme_a, condition_a, scheme_b, condition_b, question)
    ("sessions", "kmeans_K256", "sessions", "product_M2_K16", "8 bits: one alphabet vs 2 parallel"),
    ("sessions", "kmeans_K256", "sessions", "residual_16x2", "8 bits: one alphabet vs coarse-to-fine"),
    ("sessions", "residual_256x2", "sessions", "product_M2_K256", "16 bits: coarse-to-fine vs parallel"),
    ("sessions", "residual_256x3", "sessions", "product_M4_K64", "24 bits: coarse-to-fine vs parallel"),
    ("sessions", "kmeans_K1024", "sessions", "kmeans_K512", "plateau: 1024 vs 512 letters"),
    ("sessions", "pca_d16", "sessions", "residual_256x3", "16 continuous dims vs best 24-bit code"),
    ("sessions", "kmeans_K256", "sessions", "random_K256", "letter choice matters"),
    ("monkeys", "product_M2_K16", "monkeys", "kmeans_K256", "other monkey, 8 bits: parallel vs one"),
    ("monkeys", "residual_16x2", "monkeys", "kmeans_K256", "other monkey, 8 bits: coarse-to-fine vs one"),
    ("monkeys", "product_M4_K64", "monkeys", "residual_16x6", "other monkey, 24 bits: parallel vs c-to-f"),
    ("sessions", "kmeans_K256", "monkeys", "kmeans_K256", "one alphabet: other sessions vs other monkey"),
    ("sessions", "product_M4_K64", "monkeys", "product_M4_K64", "parallel: other sessions vs other monkey"),
]
DIRECTION_TESTS = [
    (sc, f"dir_{c}", sc, c, f"{sc}: direction-only vs whole vector, {c}")
    for sc in ("sessions", "monkeys")
    for c in ("kmeans_K16", "kmeans_K256", "kmeans_K1024", "product_M4_K64", "residual_16x6",
              "residual_256x3")
]


def load():
    d = pd.read_csv(R / "codebook.csv", float_precision="round_trip")
    f = R / "codebook_direction.csv"
    if f.exists():
        x = pd.read_csv(f, float_precision="round_trip")
        own = x[x.condition == "own"].set_index(["session", "scheme"]).r2
        assert np.array_equal(own.to_numpy(),
                              d[d.condition == "own"].set_index(["session", "scheme"]).r2
                              .reindex(own.index).to_numpy()), "the two runs' baselines differ"
        d = pd.concat([d, x[x.condition != "own"]], ignore_index=True)
    d["kept"] = d.r2 / d.r2_base
    return d


def boot_median(x, rng, n=1000):
    x = np.asarray(x, float)
    return np.percentile(np.median(rng.choice(x, (n, len(x))), axis=1), [2.5, 97.5])


def summary(d):
    rng = np.random.default_rng(SEED)
    rows = []
    for (sc, cond), g in d.groupby(["scheme", "condition"], sort=False):
        for m, gg in [("all", g)] + list(g.groupby("monkey")):
            lo, hi = boot_median(gg.kept, rng)
            rows.append(dict(scheme=sc, condition=cond, family=gg.family.iloc[0],
                             bits=gg.bits.iloc[0], monkey=m, n_sessions=len(gg),
                             kept_median=gg.kept.median(), kept_ci_lo=lo, kept_ci_hi=hi,
                             kept_q25=gg.kept.quantile(0.25), kept_q75=gg.kept.quantile(0.75),
                             share_r2_pos=float((gg.r2 > 0).mean()),
                             rel_err_median=gg.rel_err.median()))
    return pd.DataFrame(rows)


def tests(d):
    k = d.set_index(["scheme", "condition", "session"]).kept.sort_index()
    have = set(zip(d.scheme, d.condition))
    rows = []
    for sa, ca, sb, cb, q in TESTS + DIRECTION_TESTS:
        if (sa, ca) not in have or (sb, cb) not in have:
            continue
        a, b = k.loc[sa, ca], k.loc[sb, cb]
        a, b = a.align(b, join="inner")
        diff = (a - b).to_numpy()
        rows.append(dict(question=q, a=f"{sa}/{ca}", b=f"{sb}/{cb}", n_sessions=len(diff),
                         median_a=a.median(), median_b=b.median(),
                         median_diff=float(np.median(diff)), share_a_better=float((diff > 0).mean()),
                         wilcoxon_p=wilcoxon(diff).pvalue))
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm"] = multipletests(out.wilcoxon_p, method="holm")[1]
    return out


def embeddings(units):
    sd = torch.load(PROJECT / "checkpoints" / "poyo_mp.ckpt", map_location="cpu",
                    weights_only=False)["state_dict"]
    return sd["model.unit_emb.weight"].numpy().astype(np.float64)[units.emb_row.to_numpy()]


def electrode_agreement(units, letters, rng, n_null=200_000):
    """P(same letter) for pairs of units on the same electrode (monkey + channel) in different
    sessions, and for random pairs of the same monkey on different electrodes and sessions."""
    same, hits = 0, 0
    for _, g in units.assign(L=letters).groupby(["monkey", "channel"]):
        if g.session.nunique() < 2:
            continue
        s, L = g.session.to_numpy(), g.L.to_numpy()
        diff_sess = s[:, None] != s[None, :]
        eq = L[:, None] == L[None, :]
        iu = np.triu_indices(len(g), 1)
        same += int(diff_sess[iu].sum())
        hits += int((eq & diff_sess)[iu].sum())
    null_hits, null_n = 0, 0
    for _, g in units.assign(L=letters).groupby("monkey"):
        i, j = rng.integers(len(g), size=(2, n_null))
        s, c, L = g.session.to_numpy(), g.channel.to_numpy(), g.L.to_numpy()
        ok = (s[i] != s[j]) & (c[i] != c[j])
        null_hits += int((L[i] == L[j])[ok].sum())
        null_n += int(ok.sum())
    return same, hits / same, null_hits / null_n


def letters_table(units, X):
    """Letters fitted on all units (descriptive): what do they track?"""
    rng = np.random.default_rng(SEED)
    codes = {}
    for k in (16, 256):
        codes[f"kmeans_K{k}"] = nearest(X, fit_centroids(X, k, 0))
        D, _ = unit_rows(X)
        codes[f"dir_kmeans_K{k}"] = nearest(D, fit_centroids(D, k, 0))
    mu, vt = fit_rotation(X)
    Z = (X - mu) @ vt.T
    for j, b in enumerate(blocks(Z.shape[1], 4)):
        codes[f"product_M4_K64_block{j + 1}"] = nearest(Z[:, b], fit_centroids(Z[:, b], 64, j))
    rows = []
    for name, L in codes.items():
        n_pairs, p_same, p_null = electrode_agreement(units, L, rng)
        row = dict(letters=name, n_letters=len(np.unique(L)))
        for lab in ("monkey", "session", "area", "task"):
            row[f"ami_{lab}"] = adjusted_mutual_info_score(units[lab], L)
        row.update(same_electrode_pairs=n_pairs, p_same_letter_same_electrode=p_same,
                   p_same_letter_other_electrode=p_null)
        rows.append(row)
    return pd.DataFrame(rows)


def nearest_distance(X, src, dst_mask_fn, chunk=1024):
    """For each unit i, the Euclidean distance to the nearest unit j with dst_mask_fn(i)[j]."""
    sq = (X * X).sum(1)
    out = np.empty(len(src))
    for a in range(0, len(src), chunk):
        idx = src[a:a + chunk]
        dist = sq[idx, None] - 2 * X[idx] @ X.T + sq[None, :]
        for r, i in enumerate(idx):
            out[a + r] = np.sqrt(max(dist[r][dst_mask_fn(i)].min(), 0.0))
    return out


def stats(units, X):
    length = np.linalg.norm(X, axis=1)
    mk, ss = units.monkey.to_numpy(), units.session.to_numpy()
    res = dict(length_by_monkey={m: dict(median=float(np.median(length[mk == m])),
                                         q25=float(np.quantile(length[mk == m], 0.25)),
                                         q75=float(np.quantile(length[mk == m], 0.75)))
                                 for m in sorted(set(mk))})
    near = {}
    for m in sorted(set(mk)):
        src = np.flatnonzero(mk == m)
        same = nearest_distance(X, src, lambda i: (mk == mk[i]) & (ss != ss[i]))
        other = nearest_distance(X, src, lambda i: mk != mk[i])
        near[m] = dict(median_nearest_same_monkey_other_session=float(np.median(same)),
                       median_nearest_other_monkey=float(np.median(other)),
                       share_closer_to_own_monkey=float((same < other).mean()))
    res["nearest_unit"] = near
    return res


def main():
    d = load()
    s = summary(d)
    s.to_csv(R / "codebook_summary.csv", index=False)
    t = tests(d)
    t.to_csv(R / "codebook_tests.csv", index=False)
    units = pd.read_csv(R / "emb_units.csv")
    X = embeddings(units)
    lt = letters_table(units, X)
    lt.to_csv(R / "codebook_letters.csv", index=False)
    st = stats(units, X)
    (R / "codebook_stats.json").write_text(json.dumps(st, indent=2))

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    a = s[s.monkey == "all"].sort_values(["scheme", "family", "bits"])
    print(a[["scheme", "condition", "bits", "kept_median", "kept_ci_lo", "kept_ci_hi",
             "share_r2_pos", "rel_err_median"]].round(3).to_string(index=False))
    print(t[["question", "median_a", "median_b", "median_diff", "share_a_better",
             "wilcoxon_p_holm"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(lt.round(3).to_string(index=False))
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
