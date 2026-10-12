"""Cross-session transfer without training: evaluate POYO-MP on a session's test split with its
unit and/or session embeddings replaced by vectors from other sessions (src/transfer.py).

Donor = another session of the same monkey. The main donor is the nearest-date session of the
same monkey and task. Conditions per target session:

  own_swap              own vectors written through the swap path (must equal baseline exactly)
  own_electrode_mean    each unit <- mean of its own session's units on the same electrode (the
                        loss from averaging units per electrode, without changing day)
  matched               each unit <- mean embedding of the donor's units on the same electrode;
                        units without one <- donor population mean
  shuffled_within_area (seeds)  as matched, donor electrode labels permuted within M1 / within PMd
                        (same coverage, no match)
  donor_mean            every unit <- donor population mean (no unit identity)
  session_other_monkey  session embedding <- a same-task session of another monkey (seeded)
  matched_gap / shuffled_gap   matched / shuffled (seed 0) with one same-task donor per gap bin
                        (<= 7 d, 8-30 d, 31-180 d, > 180 d; seeded choice within the bin)

With --set matching, label-free matching instead (no behaviour labels, no training; main donor
unless noted). Unit statistics are log10 firing rate and ISI CV on the train split, standardised
over all units:
  matched                    as above (reference; must equal the main run)
  electrode_rate             on a shared electrode, the donor unit with the nearest statistics;
                             otherwise the nearest donor unit of the same brain area
  rate_only                  every unit takes the nearest donor unit of the same area (no electrodes)
  matched_3days              as matched, pooling the donor units of the 3 nearest same-task days

With --set combined (same statistics; no behaviour labels, no training):
  matched_3days_rate         pool the units of the 3 nearest same-task days, then electrode_rate
  electrode_rate_gap / matched_3days_gap / matched_3days_rate_gap   per days-apart bin, with the
                             same single donor as matched_gap; the 3-day versions pool it with the
                             2 other same-task sessions of the bin nearest to it in date

Writes results/<out>.csv (one row per session x condition x donor x seed).
Usage: python -u scripts/transfer/run_transfer.py --out dev/transfer_pilot --sessions s1 s2
       python -u scripts/transfer/run_transfer.py --set matching --out transfer_matching
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.bootstrap import PairedTrialBootstrap  # noqa: E402
from src.evaluate import (  # noqa: E402
    BRAINSET, DATA_ROOT, DEFAULT_CKPT, PROJECT, TASK, TRAIN_RECORDING_IDS, build_dataset,
    iter_sessions, load_model,
)
from src.perturb import stable_key  # noqa: E402
from src.transfer import (  # noqa: E402
    electrode_mean_vectors, permuted_channels, stat_matched_vectors, swapped_rows,
)

GAPS = [(0, 7, "<=7d"), (8, 30, "8-30d"), (31, 180, "31-180d"), (181, 10**5, ">180d")]
COLS = ["session", "task", "monkey", "n_units", "condition", "donor", "donor_days", "gap", "seed",
        "coverage", "r2", "dr2", "dr2_boot_se", "dr2_ci_lo", "dr2_ci_hi", "n_scored", "runtime_s"]


def days(a, b):
    return abs((pd.Timestamp(a[2:10]) - pd.Timestamp(b[2:10])).days)


def donors_for(s, units):
    """Main donor, other-monkey donor and one donor per gap bin."""
    meta = units.drop_duplicates("session").set_index("session")
    m, t = meta.loc[s, "monkey"], meta.loc[s, "task"]
    same = [d for d in meta.index if d != s and meta.loc[d, "monkey"] == m]
    same_task = sorted((d for d in same if meta.loc[d, "task"] == t), key=lambda d: (days(s, d), d))
    rng = np.random.default_rng([20261006, stable_key(s)])
    other_monkey = sorted(d for d in meta.index
                          if meta.loc[d, "monkey"] != m and meta.loc[d, "task"] == t)
    gaps = {}
    for lo, hi, name in GAPS:
        cand = [d for d in same_task if lo <= days(s, d) <= hi]
        if cand:
            gaps[name] = cand[rng.integers(len(cand))]
    om = other_monkey[rng.integers(len(other_monkey))] if other_monkey else None  # after gap draws
    return dict(main=same_task[0] if same_task else None, other_monkey=om, gaps=gaps)


def donor_pools(s, units, k=3):
    """Pools of donor days (lists of sessions): 'main' = the k nearest same-task sessions of the
    same monkey; per days-apart bin, the bin's donor (as in donors_for) plus the k - 1 other
    same-task sessions of the bin nearest to it in date. Used by the POYO runs and the ridge."""
    meta = units.drop_duplicates("session").set_index("session")
    same_task = [x for x in meta.index if x != s and meta.loc[x, "monkey"] == meta.loc[s, "monkey"]
                 and meta.loc[x, "task"] == meta.loc[s, "task"]]
    d = donors_for(s, units)
    pools = dict(main=sorted(same_task, key=lambda x: (days(s, x), x))[:k] if d["main"] else [])
    for lo, hi, name in GAPS:
        if name in d["gaps"]:
            dg = d["gaps"][name]
            in_bin = [x for x in same_task if lo <= days(s, x) <= hi and x != dg]
            pools[name] = [dg] + sorted(in_bin, key=lambda x: (days(dg, x), x))[:k - 1]
    return pools


def conditions(s, units, W, Ws, srow):
    """Yield (condition, donor, gap, seed, coverage, unit_rows, session_rows)."""
    tgt = units[units.session == s]
    rows, ch = tgt.emb_row.to_numpy(), tgt.channel.to_numpy()
    d = donors_for(s, units)

    def unit_swap(donor, seed=None, within_area=False):
        dn = units[units.session == donor]
        dch = dn.channel.to_numpy()
        if seed is not None:
            rng = np.random.default_rng([seed, stable_key(s), stable_key(donor)]
                                        + ([1] if within_area else []))
            dch = permuted_channels(dch, rng, dn.area.to_numpy() if within_area else None)
        dv = W[dn.emb_row.to_numpy()]
        v, matched = electrode_mean_vectors(ch, dch, dv, dv.mean(0))
        return dict(zip(rows, v)), float(matched.mean())

    yield "own_swap", None, None, -1, 1.0, {r: W[r] for r in rows}, {srow[s]: Ws[srow[s]]}
    own, _ = electrode_mean_vectors(ch, ch, W[rows], W[rows].mean(0))
    yield "own_electrode_mean", None, None, -1, 1.0, dict(zip(rows, own)), None
    if d["main"]:
        dm = d["main"]
        u, cov = unit_swap(dm)
        yield "matched", dm, None, -1, cov, u, None
        for seed in (0, 1, 2):
            us, cov_s = unit_swap(dm, seed, within_area=True)
            yield "shuffled_within_area", dm, None, seed, cov_s, us, None
        dmean = W[units[units.session == dm].emb_row.to_numpy()].mean(0)
        yield "donor_mean", dm, None, -1, 0.0, {r: dmean for r in rows}, None
    if d["other_monkey"]:
        yield ("session_other_monkey", d["other_monkey"], None, -1, np.nan, None,
               {srow[s]: Ws[srow[d["other_monkey"]]]})
    for name, dg in d["gaps"].items():
        u, cov = unit_swap(dg)
        yield "matched_gap", dg, name, -1, cov, u, None
        us, cov_s = unit_swap(dg, 0)
        yield "shuffled_gap", dg, name, 0, cov_s, us, None


FEATS = ["log10_rate_whole", "isi_cv_whole"]  # firing statistics from spikes alone


def unit_features(units):
    """Standardised label-free firing statistics per unit (aligned to `units`)."""
    props = pd.read_csv(PROJECT / "results" / "unit_properties.csv", usecols=["unit_id"] + FEATS)
    f = units[["unit_id"]].merge(props, on="unit_id", how="left")[FEATS]
    assert len(f) == len(units)
    f = f.fillna(f.median())                      # 3 silent units have no rate / ISI CV
    return ((f - f.mean()) / f.std()).to_numpy()


def matching_conditions(s, units, W, Ws, srow):
    """Label-free matching conditions (see the module docstring)."""
    tgt = units[units.session == s]
    rows, ch, area, tf = (tgt.emb_row.to_numpy(), tgt.channel.to_numpy(), tgt.area.to_numpy(),
                          tgt.feats.to_list())
    d = donors_for(s, units)
    if not d["main"]:
        return
    dm = d["main"]
    dn = units[units.session == dm]
    dch, darea, dfeat = dn.channel.to_numpy(), dn.area.to_numpy(), dn.feats.to_list()
    dv = W[dn.emb_row.to_numpy()]
    v, m = electrode_mean_vectors(ch, dch, dv, dv.mean(0))
    yield "matched", dm, None, -1, float(m.mean()), dict(zip(rows, v)), None
    for name, kw in (("electrode_rate", {}), ("rate_only", dict(use_electrode=False))):
        v, m = stat_matched_vectors(ch, area, tf, dch, darea, dfeat, dv, **kw)
        yield name, dm, None, -1, float(m.mean()), dict(zip(rows, v)), None
    pool = units[units.session.isin(donor_pools(s, units)["main"])]
    pv = W[pool.emb_row.to_numpy()]
    v, m = electrode_mean_vectors(ch, pool.channel.to_numpy(), pv, pv.mean(0))
    yield "matched_3days", dm, None, -1, float(m.mean()), dict(zip(rows, v)), None


def combined_conditions(s, units, W, Ws, srow):
    """Pooled donor days + rate matching, overall and per days-apart bin."""
    tgt = units[units.session == s]
    rows, ch, area, tf = (tgt.emb_row.to_numpy(), tgt.channel.to_numpy(), tgt.area.to_numpy(),
                          tgt.feats.to_list())
    d = donors_for(s, units)
    pools = donor_pools(s, units)

    def vectors(donors, rate):
        dn = units[units.session.isin(donors)]
        dv = W[dn.emb_row.to_numpy()]
        if rate:
            v, m = stat_matched_vectors(ch, area, tf, dn.channel.to_numpy(), dn.area.to_numpy(),
                                        dn.feats.to_list(), dv)
        else:
            v, m = electrode_mean_vectors(ch, dn.channel.to_numpy(), dv, dv.mean(0))
        return dict(zip(rows, v)), float(m.mean())

    if d["main"]:
        u, cov = vectors(pools["main"], rate=True)
        yield "matched_3days_rate", d["main"], None, -1, cov, u, None
    for (_, _, name) in GAPS:
        if name not in d["gaps"]:
            continue
        dg, pool = d["gaps"][name], pools[name]
        for cond, donors, rate in (("electrode_rate_gap", [dg], True),
                                   ("matched_3days_gap", pool, False),
                                   ("matched_3days_rate_gap", pool, True)):
            u, cov = vectors(donors, rate)
            yield cond, dg, name, -1, cov, u, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="results/<out>.csv")
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--set", choices=["main", "matching", "combined"], default="main")
    ap.add_argument("--device", default="cuda",
                    help="cpu reproduces GPU R² within ~2e-7 (scripts/audit/audit_transfer.py)")
    args = ap.parse_args()
    gen = dict(main=conditions, matching=matching_conditions, combined=combined_conditions)[args.set]
    sessions = args.sessions or sorted(TRAIN_RECORDING_IDS)
    out = PROJECT / "results" / f"{args.out}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = (pd.read_csv(out, float_precision="round_trip").to_dict("records")
            if out.exists() else [])
    def key(r):
        donor = r["donor"] if isinstance(r["donor"], str) else ""  # None / NaN = no donor
        return (r["session"], r["condition"], donor, int(r["seed"]))

    done = {key(r) for r in rows}

    units = pd.read_csv(PROJECT / "results" / "emb_units.csv")
    if args.set in ("matching", "combined"):
        units["feats"] = list(unit_features(units))
    torch.manual_seed(0)
    if args.device == "cpu":
        torch.set_num_threads(16)
    model = load_model(DEFAULT_CKPT, args.device)
    W = model.unit_emb.weight.detach().cpu().numpy()
    Ws = model.session_emb.weight.detach().cpu().numpy()
    vocab = model.unit_emb.vocab
    assert all(vocab[u] == r for u, r in zip(units.unit_id, units.emb_row)), "row map changed"
    srow = {s: model.session_emb.vocab[f"{BRAINSET}/{s}"] for s in sorted(TRAIN_RECORDING_IDS)}
    wrapped, nested = build_dataset(model, str(DATA_ROOT), sessions)
    sweep = pd.read_csv(PROJECT / "results" / "sweep.csv", float_precision="round_trip")
    stored = sweep[sweep.experiment == "baseline"].set_index("session").r2
    t_start = time.perf_counter()

    def save():
        pd.DataFrame(rows, columns=COLS).to_csv(out, index=False)

    for s in sessions:
        _, base = next(iter_sessions(model, wrapped, nested, [s], None, device=args.device))
        if args.device == "cuda":
            assert base["r2"] == stored[s], f"{s}: baseline not reproduced"
        else:
            assert abs(base["r2"] - stored[s]) < 1e-6, f"{s}: baseline not reproduced on CPU"
        boot = PairedTrialBootstrap(s, base["timestamps"], base["target"], base["interval"],
                                    base["pred"], n_boot=args.n_boot)
        meta = units[units.session == s].iloc[0]
        for cond, donor, gap, seed, cov, urows, srows in gen(s, units, W, Ws, srow):
            k = key(dict(session=s, condition=cond, donor=donor, seed=seed))
            if k in done:
                continue
            w0 = model.unit_emb.weight.detach().clone()
            ws0 = model.session_emb.weight.detach().clone()
            with swapped_rows(model, urows, srows):
                _, r = next(iter_sessions(model, wrapped, nested, [s], None, device=args.device))
            assert torch.equal(model.unit_emb.weight, w0), "unit embeddings not restored"
            assert torch.equal(model.session_emb.weight, ws0), "session embeddings not restored"
            dr2, se, lo, hi = boot.delta(r["timestamps"], r["pred"])
            if cond == "own_swap":
                assert r["r2"] == base["r2"], f"{s}: own-vector swap changed R²"
            rows.append(dict(session=s, task=TASK(s), monkey=meta.monkey,
                             n_units=int((units.session == s).sum()), condition=cond,
                             donor=donor, donor_days=days(s, donor) if donor else np.nan, gap=gap,
                             seed=seed, coverage=cov, r2=r["r2"], dr2=dr2, dr2_boot_se=se,
                             dr2_ci_lo=lo, dr2_ci_hi=hi, n_scored=r["n_scored"],
                             runtime_s=round(r["runtime_s"], 3)))
            done.add(k)
        save()
        last = [x for x in rows if x["session"] == s]
        print(f"{s:36s} base {base['r2']:.4f} | " + " ".join(
            f"{x['condition']}={x['dr2']:+.3f}" for x in last
            if x["condition"] in ("matched", "shuffled_within_area", "donor_mean", "electrode_rate",
                                  "rate_only", "matched_3days", "matched_3days_rate")
            and x["seed"] in (-1, 0)) + f" ({time.perf_counter() - t_start:.0f} s)", flush=True)

    print(f"saved {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
