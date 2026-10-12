"""Cross-day comparison for the transfer study: a fixed linear decoder moved between days.

Ridge on binned spike counts per ELECTRODE (units on one electrode summed, so a decoder trained
on one day can be applied to another), with the ridge-control protocol of the spike-timing and
missing-neuron analyses (src/binned_decoder.py): fixed 100 ms grid, K = 3 bins (300 ms of
history), trained on the train split with the POYO loss weights, penalty chosen on the valid
split, scored on the test split in the scored periods (the same test time points as POYO).

Per session: one decoder fitted on its own electrodes. Conditions per target session:
  own_day          the session's own decoder on its own test data (the reference)
  cross_main       the main donor's decoder (nearest same-task session of the same monkey, as in
                   scripts/transfer/run_transfer.py) applied to the target's test data
  cross_gap        one donor per gap bin (same donors as matched_gap in run_transfer.py)
Target spikes on electrodes the donor never recorded are dropped; donor electrodes absent on the
target day get the donor's training mean (standardised value 0), the analogue of POYO's fallback
for unmatched units.

With --pooled, the counterpart of POYO's pooled donor days (run_transfer.py --set combined): the
predictions of the donor decoders of the same pools of 3 days (run_transfer.donor_pools) are
averaged.
  cross_3days      the 3 nearest same-task sessions
  cross_3days_gap  per days-apart bin, the bin's donor and the 2 sessions of the bin nearest to it
(Rate matching has no counterpart here: the electrode decoder has no unit identity.)

Writes results/transfer_ridge.csv (or, with --pooled, results/transfer_ridge_pooled.csv).
Usage: python -u scripts/transfer/ridge_cross_day.py --workers 6 [--pooled]
"""

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_transfer import days, donor_pools, donors_for  # noqa: E402
from src.binned_decoder import (  # noqa: E402
    ALPHA_GRID, WeightedRidge, count_features, session_sets,
)
from src.bootstrap import r2_np  # noqa: E402
from src.evaluate import (  # noqa: E402
    DATA_ROOT, PROJECT, TASK, TRAIN_RECORDING_IDS, SubsetPoyoMPDataset,
)

BIN_S, K = 0.1, 3
UNITS = pd.read_csv(PROJECT / "results" / "emb_units.csv")


def load(s):
    """Spike times, electrode label per spike, and the train / valid / test sets of a session."""
    data = SubsetPoyoMPDataset(str(DATA_ROOT), [s]).get_recording(s)
    u = UNITS[UNITS.session == s].sort_values("local_index")
    assert len(u) == len(data.units.id) and (u.local_index.to_numpy() == np.arange(len(u))).all()
    ts = np.asarray(data.spikes.timestamps, np.float64)
    ch = u.channel.to_numpy()[np.asarray(data.spikes.unit_index)]
    return ts, ch, session_sets(data)


def features(ts, ch, channels, t):
    """Dense (n, len(channels) * K) electrode count features (column = lag * n + electrode);
    spikes on electrodes not in `channels` are dropped."""
    index = {c: i for i, c in enumerate(channels)}
    keep = np.array([c in index for c in ch])
    idx = np.array([index[c] for c in ch[keep]], dtype=np.int64)
    X = count_features(ts[keep], idx, len(channels), t, BIN_S, K, "grid").toarray()
    return X


def fit(s):
    """Fit the session's own electrode decoder. Returns its parameters and own-day test R²."""
    ts, ch, sets = load(s)
    channels = sorted(UNITS[UNITS.session == s].channel.unique())
    feats = {k: features(ts, ch, channels, v[0]) for k, v in sets.items()}
    t_tr, y_tr, w_tr = sets["train"]
    model = WeightedRidge().fit(sp.csr_matrix(feats["train"]), y_tr, w_tr)
    val = [r2_np(model.predict(feats["valid"], model.coef(a)), sets["valid"][1])
           for a in ALPHA_GRID]
    best = int(np.argmax(val))
    beta = model.coef(ALPHA_GRID[best])
    r2_own = r2_np(model.predict(feats["test"], beta), sets["test"][1])
    return s, dict(channels=channels, mu=model.mu, beta=beta, ybar=model.ybar, alpha_idx=best,
                   r2_own=r2_own)


def apply(decoder, ts, ch, t, channels_of_target):
    """Donor decoder on target data; donor electrodes absent on the target day get the donor's
    training mean (standardised 0)."""
    X = features(ts, ch, decoder["channels"], t)
    missing = [i for i, c in enumerate(decoder["channels"]) if c not in channels_of_target]
    n = len(decoder["channels"])
    cols = [k * n + i for k in range(K) for i in missing]
    X[:, cols] = decoder["mu"][cols]
    return X @ decoder["beta"] - decoder["mu"] @ decoder["beta"] + decoder["ybar"], len(missing) / n


def evaluate_pooled(args):
    """Averaged predictions of the decoders of each donor pool."""
    s, decoders = args
    ts, ch, sets = load(s)
    t_te, y_te = sets["test"]
    tch = set(UNITS[UNITS.session == s].channel)
    meta = UNITS[UNITS.session == s].iloc[0]
    rows = []
    for name, pool in donor_pools(s, UNITS).items():
        if not pool:
            continue
        preds, miss = zip(*(apply(decoders[x], ts, ch, t_te, tch) for x in pool))
        rows.append(dict(session=s, task=TASK(s), monkey=meta.monkey,
                         condition="cross_3days" if name == "main" else "cross_3days_gap",
                         donor=pool[0], pool="+".join(pool), n_days=len(pool),
                         gap=None if name == "main" else name,
                         donor_channels_missing=float(np.mean(miss)),
                         r2=r2_np(np.mean(preds, axis=0), y_te),
                         r2_own_day=decoders[s]["r2_own"], n_test=len(t_te)))
    return rows


def evaluate_target(args):
    s, decoders = args
    ts, ch, sets = load(s)
    t_te, y_te = sets["test"]
    tch = set(UNITS[UNITS.session == s].channel)
    units_s = UNITS[UNITS.session == s]
    d = donors_for(s, UNITS)
    meta = units_s.iloc[0]
    rows = []

    def row(cond, donor, gap, seed, r2, missing_frac):
        dch = set(UNITS[UNITS.session == donor].channel) if donor else tch
        return dict(session=s, task=TASK(s), monkey=meta.monkey, condition=cond, donor=donor,
                    donor_days=days(s, donor) if donor else np.nan, gap=gap, seed=seed,
                    unit_coverage=float(units_s.channel.isin(dch).mean()),
                    donor_channels_missing=missing_frac, r2=r2,
                    r2_own_day=decoders[s]["r2_own"], n_test=len(t_te))

    pred, miss = apply(decoders[s], ts, ch, t_te, tch)
    r2 = r2_np(pred, y_te)
    assert abs(r2 - decoders[s]["r2_own"]) < 1e-9, f"{s}: own decoder not reproduced via apply"
    rows.append(row("own_day", None, None, -1, r2, miss))
    if d["main"]:
        dm = decoders[d["main"]]
        pred, miss = apply(dm, ts, ch, t_te, tch)
        rows.append(row("cross_main", d["main"], None, -1, r2_np(pred, y_te), miss))
    for name, dg in d["gaps"].items():
        pred, miss = apply(decoders[dg], ts, ch, t_te, tch)
        rows.append(row("cross_gap", dg, name, -1, r2_np(pred, y_te), miss))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sessions", nargs="*", help="targets (default: all 99)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pooled", action="store_true")
    args = ap.parse_args()
    args.out = args.out or ("transfer_ridge_pooled" if args.pooled else "transfer_ridge")
    targets = args.sessions or sorted(TRAIN_RECORDING_IDS)
    needed = set(targets)
    for s in targets:  # every donor a target needs must be fitted too
        d = donors_for(s, UNITS)
        needed |= {x for x in [d["main"], *d["gaps"].values()] if x}
        if args.pooled:
            needed |= {x for pool in donor_pools(s, UNITS).values() for x in pool}
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        decoders = dict(pool.map(fit, sorted(needed)))
        print(f"fitted {len(decoders)} decoders ({time.perf_counter() - t0:.0f} s)", flush=True)
        at_edge = [s for s, dec in decoders.items() if dec["alpha_idx"] == len(ALPHA_GRID) - 1]
        assert not at_edge, f"penalty at the upper grid edge: {at_edge}"
        job = evaluate_pooled if args.pooled else evaluate_target
        rows = [r for rs in pool.map(job, [(s, decoders) for s in targets]) for r in rs]
    out = PROJECT / "results" / f"{args.out}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out} ({len(rows)} rows, {time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
