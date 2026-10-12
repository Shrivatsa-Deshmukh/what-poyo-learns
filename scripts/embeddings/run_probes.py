"""Embedding probes: what POYO's unit embeddings encode (CPU only).

All units recorded on one electrode (monkey x channel), on any day, are held out together:
StratifiedGroupKFold(5), stratified by monkey x task, seed 0; the inner penalty search is grouped
by electrode too (src/embedding_probes.py). Targets, measured on the training split during movement
(results/unit_properties.csv): log10 firing rate and tuning strength (ridge on rank-transformed
targets), and preferred direction (ridge on cos / sin; well-tuned units only).
Controls: shuffled targets (100 permutations at the chosen penalty), a session-mean oracle that
knows only each unit's session, and R² within sessions.

Outputs: results/emb_probe_scores.csv (score, 95% session-bootstrap CI, within-session R², oracle,
shuffled 95th percentile and p; all units and per task), results/emb_probe_preds.csv.gz
Usage: python -u scripts/embeddings/run_probes.py [--n-perm 100] [--n-boot 1000]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import embedding_probes as ep  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
TARGETS = ("log10_rate", "tuning_r2_100", "pref_dir")


def load():
    """Units with their properties (one row per unit) and their raw embedding rows."""
    ck = torch.load(PROJECT / "checkpoints/poyo_mp.ckpt", map_location="cpu", weights_only=False)
    W_all = ck["state_dict"]["model.unit_emb.weight"].numpy().astype(np.float64)
    units = pd.read_csv(PROJECT / "results/emb_units.csv")
    props = pd.read_csv(PROJECT / "results/unit_properties.csv")
    d = units.merge(props.drop(columns=["session", "local_index", "monkey", "task", "area",
                                        "electrode", "channel", "emb_row"]), on="unit_id")
    assert len(d) == 9725 and d.emb_row.is_unique
    return d, W_all[d.emb_row.values]


def targets(d):
    t = {c: d[c].values.astype(float) for c in ("log10_rate", "tuning_r2_100")}
    t["pref_dir"] = np.where(d.well_tuned.astype(bool), d.pref_dir, np.nan)
    return t


def electrode_folds(d):
    elec = (d.monkey + "/" + d.channel).values
    strat = (d.monkey + "_" + d.task).values
    folds = np.full(len(d), -1)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0)
                                .split(d, strat, groups=elec)):
        folds[te] = k
    ep.check_folds(elec, folds)  # no electrode in train and test of any fold
    assert (folds >= 0).all()
    return elec, folds


def score_rows(kind, y, pred, sess, task, n_boot, rng):
    """Score on held-out predictions, for all units and per task."""
    rows = []
    for name, m in (("all", np.ones(len(y), bool)), ("CO", task == "CO"), ("RT", task == "RT")):
        yy, pp, ss = y[m], pred[m], sess[m]
        if kind == "cont":
            base = ep.session_mean_loo(yy, ss)
            ok = ~np.isnan(base)
            ci = ep.boot_sessions(lambda i: ep.r2(yy[i], pp[i]), ss, n_boot, rng)
            rows.append(dict(subset=name, n=int(m.sum()), n_sessions=len(np.unique(ss)),
                             metric="R2", score=ep.r2(yy, pp), ci_lo=ci[0], ci_hi=ci[1],
                             within_session_r2=ep.within_session_r2(yy, pp, ss),
                             session_mean_baseline=ep.r2(yy[ok], base[ok])))
        else:
            base = ep.session_circmean_loo(yy, ss)
            ok = ~np.isnan(base)
            ci = ep.boot_sessions(lambda i: ep.circ_corr(yy[i], pp[i]), ss, n_boot, rng)
            rows.append(dict(subset=name, n=int(m.sum()), n_sessions=len(np.unique(ss)),
                             metric="circ_r", score=ep.circ_corr(yy, pp), ci_lo=ci[0],
                             ci_hi=ci[1],
                             median_err_deg=float(np.degrees(np.median(ep.ang_err(yy, pp)))),
                             session_mean_baseline=ep.circ_corr(yy[ok], base[ok]),
                             baseline_median_err_deg=float(np.degrees(np.median(
                                 ep.ang_err(yy[ok], base[ok]))))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(20261004)
    d, X = load()
    T = targets(d)
    elec, folds = electrode_folds(d)
    sess, task = d.session.values, d.task.values
    rows, preds = [], []
    for tgt in TARGETS:
        y = T[tgt]
        m = np.isfinite(y)
        if tgt == "pref_dir":
            kind, r = "dir", ep.cv_direction(X[m], y[m], elec[m], folds[m])
            null = ep.null_direction(X[m], y[m], elec[m], folds[m], r["alpha"], args.n_perm, rng)
        else:
            kind, r = "cont", ep.cv_continuous(X[m], y[m], elec[m], folds[m], rank=True)
            null = ep.null_continuous(X[m], y[m], elec[m], folds[m], r["alpha"], args.n_perm, rng)
        res = score_rows(kind, r["y"], r["pred"], sess[m], task[m], args.n_boot, rng)
        for row in res:
            row.update(target=tgt)
            if row["subset"] == "all":
                row["chance95"] = float(np.percentile(null, 95))
                row["perm_p"] = float((1 + (null >= row["score"]).sum()) / (1 + args.n_perm))
        rows += res
        preds.append(pd.DataFrame(dict(unit_id=d.unit_id.values[m], session=sess[m],
                                       task=task[m], monkey=d.monkey.values[m], target=tgt,
                                       y=r["y"], pred=r["pred"])))
        print(f"{tgt:14s} " + " ".join(f"{x['subset']}={x['score']:.3f}" for x in res), flush=True)
    cols = ["target", "subset", "metric", "n", "n_sessions", "score", "ci_lo", "ci_hi",
            "within_session_r2", "session_mean_baseline", "median_err_deg",
            "baseline_median_err_deg", "chance95", "perm_p"]
    pd.DataFrame(rows)[cols].to_csv(PROJECT / "results/emb_probe_scores.csv", index=False)
    pd.concat(preds).to_csv(PROJECT / "results/emb_probe_preds.csv.gz", index=False)


if __name__ == "__main__":
    main()
