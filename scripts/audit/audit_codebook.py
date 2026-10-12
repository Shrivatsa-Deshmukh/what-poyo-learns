"""Independent cross-checks of the neuron-letters analysis (paper, main.tex, Appendix B.3).

1. Statistics recomputed from the raw rows with plain numpy / scipy: median kept shares and two
   paired tests, against results/codebook_summary.csv and codebook_tests.csv.
2. Letters rebuilt without src/codebook.py (scikit-learn k-means, numpy SVD) for two sessions,
   with the training units taken from the stored `group` column: the 256-letter codes must equal
   the stored codes, and the coding errors of the k-means, 4-parallel-letter and 16-component codes
   must equal the stored rel_err.
3. Model re-runs on the CPU in full precision (normal evaluation path, no cached batches, no fp16)
   for the same two sessions, with independently built vectors; the share kept must match the
   stored fp16 rows within 2e-3.
4. Every number of Table B4 re-derived from the summary and found in the paper (main.tex).
5. The descriptive numbers of B.3 recomputed independently: 256 letters fitted on all units
   (scikit-learn), their adjusted mutual information with session, and the same-electrode sharing
   rate (pandas self-join); embedding lengths by monkey; nearest-unit shares (scikit-learn
   NearestNeighbors), against results/codebook_letters.csv and codebook_stats.json.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.neighbors import NearestNeighbors

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from src.evaluate import DATA_ROOT, DEFAULT_CKPT, build_dataset, iter_sessions, load_model  # noqa: E402
from src.transfer import swapped_rows  # noqa: E402

R = PROJECT / "results"
SESSIONS = ["m_20140203_center_out_reaching", "c_20131009_random_target_reaching"]


def rows():
    d = pd.concat([pd.read_csv(R / "codebook.csv", float_precision="round_trip"),
                   pd.read_csv(R / "codebook_direction.csv", float_precision="round_trip")
                   .query("condition != 'own'")])
    d["kept"] = d.r2 / d.r2_base
    return d


def check_statistics(d):
    print("=== (1) statistics recomputed from raw rows")
    summ = pd.read_csv(R / "codebook_summary.csv").query("monkey == 'all'")
    summ = summ.set_index(["scheme", "condition"]).kept_median
    for sc, c in (("sessions", "kmeans_K256"), ("sessions", "residual_256x3"),
                  ("sessions", "pca_d16"), ("monkeys", "kmeans_K256"),
                  ("monkeys", "product_M4_K64"), ("monkeys", "mean_length")):
        mine = float(np.median(d[(d.scheme == sc) & (d.condition == c)].kept))
        print(f"  {sc:8s} {c:16s} median kept {mine:.6f} | stored {summ[(sc, c)]:.6f}")
        assert abs(mine - summ[(sc, c)]) < 1e-12
    t = pd.read_csv(R / "codebook_tests.csv").set_index("question")
    for q, (sa, ca, sb, cb) in {
            "8 bits: one alphabet vs 2 parallel": ("sessions", "kmeans_K256", "sessions",
                                                   "product_M2_K16"),
            "one alphabet: other sessions vs other monkey": ("sessions", "kmeans_K256",
                                                             "monkeys", "kmeans_K256")}.items():
        a = d[(d.scheme == sa) & (d.condition == ca)].set_index("session").kept
        b = d[(d.scheme == sb) & (d.condition == cb)].set_index("session").kept
        p = wilcoxon((a - b.loc[a.index]).to_numpy()).pvalue
        print(f"  {q}: p {p:.3e} | stored {t.loc[q, 'wilcoxon_p']:.3e}")
        assert np.isclose(p, t.loc[q, "wilcoxon_p"], rtol=1e-9)


def table():
    units = pd.read_csv(R / "emb_units.csv")
    sd = torch.load(DEFAULT_CKPT, map_location="cpu", weights_only=False)["state_dict"]
    return units, sd["model.unit_emb.weight"].numpy().astype(np.float64)


_CACHE = {}


def independent_codes(units, W, d, s, scheme):
    """Training / held-out units from the stored group column; codes built from scratch."""
    if (s, scheme) not in _CACHE:
        _CACHE[s, scheme] = _independent_codes(units, W, d, s, scheme)
    return _CACHE[s, scheme]


def _independent_codes(units, W, d, s, scheme):
    group = d[(d.scheme == scheme)].drop_duplicates("session").set_index("session").group
    g = group[s]
    train = units[units.session.map(group) != g]
    test = units[units.session.map(group) == g]
    tgt = units[units.session == s]
    Xtr, Xs = W[train.emb_row], W[tgt.emb_row]
    out = {}
    km = KMeans(n_clusters=256, n_init=3, random_state=0).fit(Xtr)
    dist = ((Xs[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1)
    out["kmeans_K256"] = (km.cluster_centers_[dist.argmin(1)], dist.argmin(1))
    mu = Xtr.mean(0)
    vt = np.linalg.svd(Xtr - mu, full_matrices=True)[2]
    Ztr, Zs = (Xtr - mu) @ vt.T, (Xs - mu) @ vt.T
    rec = np.empty_like(Zs)
    for j in range(4):
        b = np.arange(j, 128, 4)
        kmj = KMeans(n_clusters=64, n_init=3, random_state=j).fit(Ztr[:, b])
        dj = ((Zs[:, b][:, None, :] - kmj.cluster_centers_[None]) ** 2).sum(-1)
        rec[:, b] = kmj.cluster_centers_[dj.argmin(1)]
    out["product_M4_K64"] = (rec @ vt + mu, None)
    out["pca_d16"] = ((Xs - mu) @ vt[:16].T @ vt[:16] + mu, None)
    n = np.linalg.norm(Xs, axis=1, keepdims=True)
    out["mean_length"] = (Xs / n * np.linalg.norm(Xtr, axis=1).mean(), None)
    assert set(tgt.unit_id) <= set(test.unit_id) and not set(tgt.unit_id) & set(train.unit_id)
    return tgt, Xs, out


def check_codes(d, units, W):
    print("=== (2) letters rebuilt without src/codebook.py")
    codes = pd.read_csv(R / "codebook_codes.csv.gz")
    mu_all = W[units.emb_row].mean(0)
    for s in SESSIONS:
        for scheme in ("sessions", "monkeys"):
            tgt, Xs, out = independent_codes(units, W, d, s, scheme)
            stored = codes[(codes.scheme == scheme) & (codes.condition == "kmeans_K256")]
            stored = stored.set_index("unit_id").code.astype(int).loc[tgt.unit_id].to_numpy()
            same = (stored == out["kmeans_K256"][1]).mean()
            print(f"  {s[:24]} {scheme:8s} 256-letter codes equal: {same:.3f}")
            assert same == 1.0
            for c in ("kmeans_K256", "product_M4_K64", "pca_d16"):
                v = out[c][0]
                err = ((Xs - v) ** 2).sum() / ((Xs - mu_all) ** 2).sum()
                row = d[(d.session == s) & (d.scheme == scheme) & (d.condition == c)].iloc[0]
                print(f"    {c:16s} rel_err {err:.6f} | stored {row.rel_err:.6f}")
                assert abs(err - row.rel_err) < 1e-6


def check_reruns(d, units, W):
    print("=== (3) CPU full-precision re-runs vs stored fp16 rows")
    torch.manual_seed(0)
    torch.set_num_threads(16)
    model = load_model(DEFAULT_CKPT, "cpu")
    wrapped, nested = build_dataset(model, str(DATA_ROOT), SESSIONS)
    worst = 0.0
    for s in SESSIONS:
        _, base = next(iter_sessions(model, wrapped, nested, [s], None, device="cpu"))
        for scheme, c in (("sessions", "kmeans_K256"), ("monkeys", "kmeans_K256"),
                          ("monkeys", "product_M4_K64"), ("sessions", "pca_d16"),
                          ("monkeys", "mean_length")):
            tgt, _, out = independent_codes(units, W, d, s, scheme)
            with swapped_rows(model, dict(zip(tgt.emb_row, out[c][0]))):
                _, r = next(iter_sessions(model, wrapped, nested, [s], None, device="cpu"))
            row = d[(d.session == s) & (d.scheme == scheme) & (d.condition == c)].iloc[0]
            mine = r["r2"] / base["r2"]
            worst = max(worst, abs(mine - row.kept))
            print(f"  {s[:24]} {scheme:8s} {c:16s} kept CPU fp32 {mine:.5f} | GPU fp16 "
                  f"{row.kept:.5f}")
    print(f"  max |difference| = {worst:.1e}")
    assert worst < 2e-3


def check_report():
    print("=== (4) Table B4 of main.tex re-derived from the summary")
    summ = pd.read_csv(R / "codebook_summary.csv").query("monkey == 'all'")
    summ = summ.set_index(["scheme", "condition"])
    text = (PROJECT / "main.tex").read_text()
    lines = {}
    for line in text.splitlines():            # LaTeX table rows: "label & bits & cell & cell \\"
        if line.count("&") == 3 and line.rstrip().endswith("\\\\"):
            cells = [c.strip() for c in line.rstrip()[:-2].split("&")]
            lines[cells[0].replace("$\\times$", "×")] = cells
    table = {"one alphabet, K = 16": "kmeans_K16", "one alphabet, K = 256": "kmeans_K256",
             "one alphabet, K = 1024": "kmeans_K1024", "2 parallel × 16 letters": "product_M2_K16",
             "coarse-to-fine, 2 × 256": "residual_256x2", "4 parallel × 64 letters": "product_M4_K64",
             "coarse-to-fine, 6 × 16": "residual_16x6", "coarse-to-fine, 3 × 256": "residual_256x3",
             "top 8 principal components": "pca_d8", "top 16 principal components": "pca_d16"}
    n = 0
    for label, c in table.items():
        cells = [re.sub(r"\\textbf\{([^}]*)\}", r"\1", x) for x in lines[label][2:4]]
        for sc, cell in zip(("sessions", "monkeys"), cells):
            r = summ.loc[(sc, c)]
            want = f"{r.kept_median:.2f} [{r.kept_ci_lo:.2f}, {r.kept_ci_hi:.2f}]"
            assert cell == want, f"{label} / {sc}: report '{cell}', summary '{want}'"
            n += 1
    print(f"  {n} table cells match the summary")


def check_descriptive(units, W):
    print("=== (5) descriptive letters, lengths and nearest units recomputed")
    X = W[units.emb_row]
    L = KMeans(n_clusters=256, n_init=3, random_state=0).fit(X).labels_
    lt = pd.read_csv(R / "codebook_letters.csv").set_index("letters").loc["kmeans_K256"]
    ami = adjusted_mutual_info_score(units.session, L)
    print(f"  AMI(letter, session) {ami:.4f} | stored {lt.ami_session:.4f}")
    assert abs(ami - lt.ami_session) < 1e-9
    u = units.assign(L=L)[["monkey", "channel", "session", "L"]].reset_index()
    pairs = u.merge(u, on=["monkey", "channel"])
    pairs = pairs[(pairs.index_x < pairs.index_y) & (pairs.session_x != pairs.session_y)]
    share = (pairs.L_x == pairs.L_y).mean()
    print(f"  same-electrode pairs {len(pairs)} share a letter {share:.4f} | stored "
          f"{int(lt.same_electrode_pairs)} {lt.p_same_letter_same_electrode:.4f}")
    assert len(pairs) == lt.same_electrode_pairs and abs(share - lt.p_same_letter_same_electrode) < 1e-12
    st = json.loads((R / "codebook_stats.json").read_text())
    length = np.linalg.norm(X, axis=1)
    for m in ("C", "J", "M"):
        sel = (units.monkey == m).to_numpy()
        med = float(np.median(length[sel]))
        own = X[sel]
        other = X[~sel]
        d_other = NearestNeighbors(n_neighbors=1).fit(other).kneighbors(own)[0][:, 0]
        d_same = np.empty(sel.sum())
        sess = units.session[sel].to_numpy()
        for s_ in np.unique(sess):
            a, b = own[sess == s_], own[sess != s_]
            d_same[sess == s_] = NearestNeighbors(n_neighbors=1).fit(b).kneighbors(a)[0][:, 0]
        closer = float((d_same < d_other).mean())
        ref = st["nearest_unit"][m]["share_closer_to_own_monkey"]
        print(f"  monkey {m}: median length {med:.4f} | stored "
              f"{st['length_by_monkey'][m]['median']:.4f}; closer to own monkey {closer:.4f} | "
              f"stored {ref:.4f}")
        assert abs(med - st["length_by_monkey"][m]["median"]) < 1e-9 and abs(closer - ref) < 1e-3


if __name__ == "__main__":
    d = rows()
    check_statistics(d)
    check_report()
    units, W = table()
    check_descriptive(units, W)
    check_codes(d, units, W)
    check_reruns(d, units, W)
    print("all codebook checks passed")
