"""Neuron letters without training: evaluate POYO-MP on a session's test split with every unit
embedding of that session replaced by its code in a shared codebook (src/codebook.py).

The codebook never sees the session's own units. Two fitting schemes:
  sessions   5 folds of sessions (dealt within each monkey); a session is coded by the codebook
             fitted on the units of the other 4 folds (other days of the same monkeys included)
  monkeys    leave one monkey out; a session is coded by the codebook of the other monkeys

Conditions (unit rows only; the session embedding is the session's own):
  own                 own vectors through the swap path (must equal the baseline exactly)
  kmeans_K<k>         one alphabet of k letters (k = 1: every unit gets the training mean)
  product_M<m>_K<k>   parallel letters: m blocks of the PCA-rotated vector, k letters each
  residual_<k>x<n>    coarse-to-fine letters: n stages of k letters
  pca_d<d>            continuous control: top d principal components
  area                letter = brain area (M1 / PMd) of the unit
  random_K<k>         control: kmeans_K<k> letters permuted among the session's units (seed 0),
                      so letter usage is kept but each unit gets another unit's letter

With --set direction (is the length of the embedding what letters miss?):
  own                 as above
  dir_kmeans_K<k>, dir_product_M4_K64, dir_residual_<k>x<n>
                      letters fitted on directions only (unit vectors); each unit keeps its own
                      length (one number per unit on top of the code)
  mean_length         own direction, length = mean length of the training units (length removed)

Speed (default --precision fp16): each session's test windows are tokenized once and re-scored
for every condition, the forward pass runs under fp16 autocast, and residual_<k>x1 rows are copied
from kmeans_K<k> (identical codes). The fp32 baseline must equal the stored one exactly and the
fp16 baseline must be within 1e-3 of it; every row records both (r2_base, r2_base_fp32), and
"kept" ratios use the baseline of the same precision.

Writes results/<out>.csv (one row per session x scheme x condition) and, for the main set only,
results/<out>_codes.csv.gz (the letters of every coded unit; one row per unit x scheme x condition).
Usage: python -u scripts/codebook/run_codebook.py --out dev/codebook_pilot --sessions s1 s2
       python -u scripts/codebook/run_codebook.py --out codebook
       python -u scripts/codebook/run_codebook.py --set direction --out codebook_direction
"""

import argparse
import faulthandler
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.bootstrap import PairedTrialBootstrap  # noqa: E402
from src.codebook import (  # noqa: E402
    bits, group_means, kmeans_quantize, pca_truncate, product_quantize, residual_quantize,
    on_direction, session_folds, unit_rows,
)
from src.evaluate import (  # noqa: E402
    DATA_ROOT, DEFAULT_CKPT, PROJECT, TASK, TRAIN_RECORDING_IDS, build_dataset, evaluate_batches,
    iter_sessions, load_model, session_batches,
)
from src.perturb import stable_key  # noqa: E402
from src.transfer import swapped_rows  # noqa: E402

KMEANS_K = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
PRODUCT = [(2, 16), (2, 64), (2, 256), (4, 4), (4, 8), (4, 16), (4, 64), (8, 2), (8, 4), (8, 8)]
RESIDUAL = [(16, n) for n in range(1, 7)] + [(256, n) for n in range(1, 4)]
PCA_D = [1, 2, 4, 8, 16, 32, 64]
RANDOM_K = [16, 256]
DIR_KMEANS_K = [1, 16, 64, 256, 1024]
DIR_PRODUCT = [(4, 64)]
DIR_RESIDUAL = [(16, 6), (256, 3)]
DUPLICATE = {f"residual_{k}x1": f"kmeans_K{k}" for k, n in RESIDUAL if n == 1}
COLS = ["session", "task", "monkey", "n_units", "scheme", "group", "condition", "family", "bits",
        "rel_err", "precision", "r2_base", "r2_base_fp32", "r2", "dr2", "dr2_boot_se", "dr2_ci_lo", "dr2_ci_hi", "n_scored", "runtime_s"]


def condition_specs(cset="main"):
    """[(condition, family, bits)] in run order."""
    specs = [("own", "own", np.nan)]
    if cset == "direction":
        specs += [(f"dir_kmeans_K{k}", "dir_kmeans", bits("kmeans", k)) for k in DIR_KMEANS_K]
        specs += [(f"dir_product_M{m}_K{k}", "dir_product", bits("product", k, m))
                  for m, k in DIR_PRODUCT]
        specs += [(f"dir_residual_{k}x{n}", "dir_residual", bits("residual", k, n))
                  for k, n in DIR_RESIDUAL]
        specs += [("mean_length", "mean_length", np.nan)]
        return specs
    specs += [(f"kmeans_K{k}", "kmeans", bits("kmeans", k)) for k in KMEANS_K]
    specs += [(f"product_M{m}_K{k}", "product", bits("product", k, m)) for m, k in PRODUCT]
    specs += [(f"residual_{k}x{n}", "residual", bits("residual", k, n)) for k, n in RESIDUAL]
    specs += [(f"pca_d{d}", "pca", np.nan) for d in PCA_D]
    specs += [("area", "area", 1.0)]
    specs += [(f"random_K{k}", "random", bits("kmeans", k)) for k in RANDOM_K]
    return specs


def groups_for(scheme, units, seed=0):
    """{session: group label} for a fitting scheme."""
    meta = units.drop_duplicates("session")
    if scheme == "sessions":
        f = session_folds(meta.session.to_numpy(), meta.monkey.to_numpy(), 5, seed)
        return {s: f"fold{g}" for s, g in f.items()}
    return dict(zip(meta.session, "monkey_" + meta.monkey))


def code_group(train, test, W, seed=0, cset="main"):
    """All conditions for the held-out units `test` (rows of the unit table), codebooks fitted on
    `train`. Returns {condition: (vectors (n_test, dim), codes (n_test, ...) or None)}."""
    Xtr, Xte = W[train.emb_row.to_numpy()], W[test.emb_row.to_numpy()]
    out = {"own": (Xte.astype(np.float64), None)}
    if cset == "direction":
        for k in DIR_KMEANS_K:
            out[f"dir_kmeans_K{k}"] = on_direction(kmeans_quantize, Xtr, Xte, k, seed)
        for m, k in DIR_PRODUCT:
            out[f"dir_product_M{m}_K{k}"] = on_direction(product_quantize, Xtr, Xte, m, k, seed)
        for k, n in DIR_RESIDUAL:
            out[f"dir_residual_{k}x{n}"] = on_direction(residual_quantize, Xtr, Xte, [k] * n,
                                                        seed)
        d, _ = unit_rows(Xte)
        out["mean_length"] = (d * unit_rows(Xtr)[1].mean(), None)
        return out
    for k in KMEANS_K:
        out[f"kmeans_K{k}"] = kmeans_quantize(Xtr, Xte, k, seed)
    for m, k in PRODUCT:
        out[f"product_M{m}_K{k}"] = product_quantize(Xtr, Xte, m, k, seed)
    for k, n in RESIDUAL:
        out[f"residual_{k}x{n}"] = residual_quantize(Xtr, Xte, [k] * n, seed)
    for d in PCA_D:
        out[f"pca_d{d}"] = (pca_truncate(Xtr, Xte, d), None)
    out["area"] = group_means(Xtr, train.area.to_numpy(), test.area.to_numpy())
    for k in RANDOM_K:
        v, c = out[f"kmeans_K{k}"]
        perm = np.arange(len(test))
        sess = test.session.to_numpy()
        for s in np.unique(sess):           # permute letters among the units of each session
            idx = np.flatnonzero(sess == s)
            rng = np.random.default_rng([seed, stable_key(s), k])
            perm[idx] = idx[rng.permutation(len(idx))]
        out[f"random_K{k}"] = (v[perm], c[perm])
    return out


def rel_err(x, xhat, mu):
    """Squared coding error relative to the spread of the vectors around the global mean."""
    return float(((x - xhat) ** 2).sum() / ((x - mu) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="results/<out>.csv")
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--schemes", nargs="*", default=["sessions", "monkeys"],
                    choices=["sessions", "monkeys"])
    ap.add_argument("--set", dest="cset", choices=["main", "direction"], default="main")
    ap.add_argument("--seed", type=int, default=0, help="k-means and fold seed")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--watchdog", type=float, default=600,
                    help="seconds per session before dumping all stacks and exiting (0 = off); "
                         "the run resumes from the saved rows when restarted")
    args = ap.parse_args()
    sessions = args.sessions or sorted(TRAIN_RECORDING_IDS)
    out = PROJECT / "results" / f"{args.out}.csv"
    out_codes = PROJECT / "results" / f"{args.out}_codes.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = (pd.read_csv(out, float_precision="round_trip").to_dict("records")
            if out.exists() else [])
    done = {(r["session"], r["scheme"], r["condition"]) for r in rows}
    code_rows = pd.read_csv(out_codes).to_dict("records") if out_codes.exists() else []

    units = pd.read_csv(PROJECT / "results" / "emb_units.csv")
    torch.manual_seed(0)
    model = load_model(DEFAULT_CKPT, args.device)
    W = model.unit_emb.weight.detach().cpu().numpy()
    vocab = model.unit_emb.vocab
    assert all(vocab[u] == r for u, r in zip(units.unit_id, units.emb_row)), "row map changed"
    mu = W[units.emb_row.to_numpy()].astype(np.float64).mean(0)
    specs = condition_specs(args.cset)

    # codebooks: one fit per (scheme, group) that holds a requested session (CPU, before decoding)
    t0 = time.perf_counter()
    coded = {}                                    # (scheme, session) -> {condition: (vecs, codes)}
    for scheme in args.schemes:
        grp = groups_for(scheme, units, args.seed)
        for g in sorted({grp[s] for s in sessions}):
            in_g = units.session.map(grp) == g
            res = code_group(units[~in_g], units[in_g], W, args.seed, args.cset)
            test = units[in_g].reset_index(drop=True)
            for s in sessions:
                if grp[s] == g:
                    idx = np.flatnonzero(test.session == s)
                    coded[scheme, s] = {c: (v[idx], None if k is None else k[idx], g)
                                        for c, (v, k) in res.items()}
            print(f"codebooks {scheme}/{g}: {in_g.sum()} held-out units "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)

    have_codes = {(r["unit_id"], r["scheme"], r["condition"]) for r in code_rows}
    for (scheme, s), conds in (coded.items() if args.cset == "main" else []):
        uid = units[units.session == s].unit_id.to_numpy()
        for c, (_, k, _) in conds.items():
            if k is None:
                continue
            k2 = k.reshape(len(k), -1)
            for u, kk in zip(uid, k2):
                if (u, scheme, c) not in have_codes:
                    code_rows.append(dict(unit_id=u, scheme=scheme, condition=c,
                                          code="-".join(map(str, kk))))
    if args.cset == "main":
        pd.DataFrame(code_rows, columns=["unit_id", "scheme", "condition", "code"]).to_csv(
            out_codes, index=False)

    wrapped, nested = build_dataset(model, str(DATA_ROOT), sessions)
    sweep = pd.read_csv(PROJECT / "results" / "sweep.csv", float_precision="round_trip")
    stored = sweep[sweep.experiment == "baseline"].set_index("session").r2
    t_start = time.perf_counter()

    amp = args.precision == "fp16"
    for s in sessions:
        if args.watchdog:   # a stall (GPU idle, one CPU thread spinning, >1 h) was seen once
            faulthandler.dump_traceback_later(args.watchdog, exit=True)
        _, base32 = next(iter_sessions(model, wrapped, nested, [s], None, device=args.device))
        if args.device == "cuda":
            assert base32["r2"] == stored[s], f"{s}: baseline not reproduced"
        else:
            assert abs(base32["r2"] - stored[s]) < 1e-6, f"{s}: baseline not reproduced on CPU"
        batches, interval = session_batches(model, wrapped, nested, s, device=args.device)
        base = evaluate_batches(model, batches, s, interval, args.device, amp)
        if amp:
            assert abs(base["r2"] - base32["r2"]) < 1e-3, f"{s}: fp16 baseline too far from fp32"
        else:
            assert base["r2"] == base32["r2"], f"{s}: cached batches changed the baseline"
        boot = PairedTrialBootstrap(s, base["timestamps"], base["target"], base["interval"],
                                    base["pred"], n_boot=args.n_boot)
        tgt = units[units.session == s]
        rws = tgt.emb_row.to_numpy()
        x = W[rws].astype(np.float64)
        for scheme in args.schemes:
            scored = {}
            for cond, fam, b in specs:
                v, k, g = coded[scheme, s][cond]
                if cond in DUPLICATE:          # same codes as the k-means row: copy its score
                    v0, k0, _ = coded[scheme, s][DUPLICATE[cond]]
                    # multithreaded k-means is reproducible only to ~1e-16; the letters are equal
                    assert np.array_equal(k[:, 0], k0) and np.allclose(v, v0, rtol=0, atol=1e-9)
                if (s, scheme, cond) in done:
                    continue
                if cond in DUPLICATE and DUPLICATE[cond] in scored:
                    r = scored[DUPLICATE[cond]]
                else:
                    w0 = model.unit_emb.weight.detach().clone()
                    with swapped_rows(model, dict(zip(rws, v))):
                        r = evaluate_batches(model, batches, s, interval, args.device, amp)
                    assert torch.equal(model.unit_emb.weight, w0), "unit embeddings not restored"
                    scored[cond] = r
                if cond == "own":
                    assert r["r2"] == base["r2"], f"{s}: own-vector swap changed R²"
                dr2, se, lo, hi = boot.delta(r["timestamps"], r["pred"])
                rows.append(dict(session=s, task=TASK(s), monkey=tgt.monkey.iloc[0],
                                 n_units=len(tgt), scheme=scheme, group=g, condition=cond,
                                 family=fam, bits=b, rel_err=rel_err(x, v, mu),
                                 precision=args.precision, r2_base=base["r2"],
                                 r2_base_fp32=base32["r2"], r2=r["r2"],
                                 dr2=dr2, dr2_boot_se=se, dr2_ci_lo=lo, dr2_ci_hi=hi,
                                 n_scored=r["n_scored"], runtime_s=round(r["runtime_s"], 3)))
                done.add((s, scheme, cond))
        pd.DataFrame(rows, columns=COLS).to_csv(out, index=False)
        last = {(x_["scheme"], x_["condition"]): x_["r2"] / base["r2"]
                for x_ in rows if x_["session"] == s}
        show = (["kmeans_K16", "kmeans_K256", "product_M4_K16", "residual_16x2", "area"]
                if args.cset == "main" else ["dir_kmeans_K256", "dir_product_M4_K64",
                                             "mean_length"])
        print(f"{s:36s} base {base['r2']:.4f} | kept " + " ".join(
            f"{c}={last.get(('sessions', c), np.nan):.2f}" for c in show)
            + f" ({time.perf_counter() - t_start:.0f} s)", flush=True)

    faulthandler.cancel_dump_traceback_later()
    print(f"saved {out} ({len(rows)} rows)"
          + (f", {out_codes} ({len(code_rows)} rows)" if args.cset == "main" else ""))


if __name__ == "__main__":
    main()
