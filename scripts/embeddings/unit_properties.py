"""Unit properties driver: per-unit firing and tuning properties on the training split.

Main columns: training split during movement, i.e. 20 ms bins whose speed exceeds the 20th
percentile of the speed in train_domain ∩ readout eval periods (CO reach / RT random-target), per
session (threshold from training data only). *_whole columns: firing statistics (spike count, rate,
ISI CV) over the whole training split, used to match units across days by firing alone and for the
spike-count control of the rate probe.
Tuning strength: tuning_r2_100 = CV R² at 100 ms bins, refitted at the lag chosen on the 20 ms fit
(no second lag search); tuning_r2 (20 ms) is kept as a column.
Flags: low_spikes (< 200 spikes), silent_train (0 spikes during movement -> tuning columns NaN),
at_lag_edge (best lag at ±300 ms), well_tuned (20 ms CV R² above the unit's 95th percentile of
the lag-search-aware shifted-velocity null).

Output: results/unit_properties<suffix>.csv, one row per unit. Pilot runs (any --suffix) write
to results/dev/ instead of results/.
Usage: python scripts/embeddings/unit_properties.py --sessions s1 s2 --suffix _pilot
       python scripts/embeddings/unit_properties.py --workers 6            (all 99 sessions)
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "3")  # before numpy: several sessions run in parallel
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluate import DATA_ROOT, PROJECT, TRAIN_RECORDING_IDS, SubsetPoyoMPDataset  # noqa: E402
from src.perturb import stable_key  # noqa: E402
from src import unit_properties as up  # noqa: E402
from torch_brain.data import Interval  # noqa: E402

N_SHIFTS = 100
MIN_SPIKES = 200
LAG_EDGE_MS = 300
TUNING_COLS = ["best_lag_ms", "tuning_r2", "tuning_r2_100", "speed_model_r2", "b0", "bx", "by",
               "tuning_depth", "tuning_depth_norm", "pref_dir", "null_r2_95", "tuning_p"]


def plain(iv):
    return Interval(start=np.asarray(iv.start, np.float64), end=np.asarray(iv.end, np.float64))


def scope_properties(st, su, n_units, rate, vel, edges, iv, heldout, rng):
    f = up.firing_stats(st, su, n_units, iv)
    if heldout is not None:
        assert not up.in_intervals(st[f["inside"]], heldout).any(), "held-out spike counted"
    j = up.scored_bins(edges, iv, vel, forbidden=heldout)
    t = up.tuning(rate[j], vel, j, n_shifts=N_SHIFTS, rng=rng)
    silent_bins = rate[j].sum(0) == 0
    assert np.array_equal(np.isnan(t["tuning_r2"]), silent_bins), "NaN R² for a non-silent unit"
    thr = np.full(n_units, np.nan)
    p = np.full(n_units, np.nan)
    ok = ~silent_bins
    thr[ok] = np.percentile(t["null"][:, ok], 95, axis=0)
    p[ok] = (1 + (t["null"][:, ok] >= t["tuning_r2"][None, ok]).sum(0)) / (1 + N_SHIFTS)
    lag_bins = (t["best_lag_ms"] / (up.BIN_S * 1000)).round().astype(int)
    r2_100, n100 = up.coarse_cv_r2(rate, vel, j, lag_bins)
    depth = np.hypot(t["bx"], t["by"])
    mean_rate = rate[j].mean(0)
    silent = f["spike_count"] == 0
    df = pd.DataFrame(dict(
        spike_count=f["spike_count"], rate_hz=f["rate_hz"], log10_rate=f["log10_rate"],
        isi_cv=f["isi_cv"], best_lag_ms=t["best_lag_ms"], tuning_r2=t["tuning_r2"],
        tuning_r2_100=r2_100, speed_model_r2=t["speed_model_r2"], b0=t["b0"], bx=t["bx"],
        by=t["by"], tuning_depth=depth,
        tuning_depth_norm=np.where(mean_rate > 0, depth / np.where(mean_rate > 0, mean_rate, 1),
                                   np.nan),
        pref_dir=np.arctan2(t["by"], t["bx"]), null_r2_95=thr, tuning_p=p,
        low_spikes=f["spike_count"] < MIN_SPIKES, silent_train=silent,
        at_lag_edge=np.abs(t["best_lag_ms"]) >= LAG_EDGE_MS,
        well_tuned=np.where(silent_bins, False, t["tuning_r2"] > np.nan_to_num(thr, nan=np.inf)),
        used_s=up.total_duration(iv), n_scored_bins=len(j), n_scored_bins_100=n100))
    # silent units: no tuning targets (kept for the embedding-norm analysis)
    df.loc[df.silent_train | silent_bins, TUNING_COLS] = np.nan
    df.loc[df.silent_train | silent_bins, "well_tuned"] = False
    return df, lag_bins


def run_session(s):
    t0 = time.perf_counter()
    data = SubsetPoyoMPDataset(str(DATA_ROOT), [s]).get_recording(s)  # adds readout_config
    n_units = len(data.units.id)
    st = np.asarray(data.spikes.timestamps)
    su = np.asarray(data.spikes.unit_index)
    heldout = plain(data.test_domain) | plain(data.valid_domain)
    rate, vel, edges = up.bin_recording(data)
    tr_dom = plain(data.train_domain)
    mov, thr = up.moving_intervals(edges, vel, up.used_intervals(data, "train", "scored"))
    rng = np.random.default_rng([20261003, stable_key(s), 0])
    out, _ = scope_properties(st, su, n_units, rate, vel, edges, tr_dom & mov, heldout, rng)
    whole = up.firing_stats(st, su, n_units, tr_dom)
    for k in ("spike_count", "rate_hz", "log10_rate", "isi_cv"):
        out[f"{k}_whole"] = whole[k]
    out["used_s_whole"] = up.total_duration(tr_dom)
    out["speed_threshold"] = thr
    out["frac_train_time_kept"] = out["used_s"] / out["used_s_whole"]
    out.insert(0, "local_index", np.arange(n_units))
    out.insert(0, "session", s)
    return s, out, round(time.perf_counter() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    sessions = args.sessions or sorted(TRAIN_RECORDING_IDS)
    units = pd.read_csv(PROJECT / "results/emb_units.csv")
    trains = []
    t_start = time.perf_counter()
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            results = list(ex.map(run_session, sessions))
    else:
        results = [run_session(s) for s in sessions]
    for s, tr, dt in results:
        meta = units[units.session == s].set_index("local_index")
        assert len(meta) == len(tr), s
        for c in ("unit_id", "monkey", "task", "area", "electrode", "channel", "emb_row"):
            tr[c] = meta.loc[tr.local_index, c].values
        trains.append(tr)
        print(f"{s:36s} units={len(tr):3d} moving={tr.used_s.iloc[0]:6.1f}s "
              f"({tr.frac_train_time_kept.iloc[0]:.0%} of train) "
              f"whole={tr.used_s_whole.iloc[0]:6.1f}s well_tuned={tr.well_tuned.mean():.0%} ({dt} s)",
              flush=True)
    train = pd.concat(trains, ignore_index=True)
    assert train.unit_id.is_unique and train.emb_row.is_unique
    # documented NaNs only: log10_rate (rate 0), isi_cv (< 3 spikes), tuning columns (silent)
    nan_ok = {"log10_rate", "isi_cv", "log10_rate_whole", "isi_cv_whole", *TUNING_COLS}
    bad = [c for c in train.columns if train[c].isna().any() and c not in nan_ok]
    assert not bad, f"unexpected NaNs in {bad}"
    sil = train.silent_train
    assert train.loc[~sil, "tuning_r2"].notna().all(), "non-silent unit without tuning R²"
    res = PROJECT / "results" / ("dev" if args.suffix else "")  # pilot runs go to results/dev/
    res.mkdir(parents=True, exist_ok=True)
    train.to_csv(res / f"unit_properties{args.suffix}.csv", index=False)
    print("saved", f"results/unit_properties{args.suffix}.csv", len(train), "units",
          f"({round(time.perf_counter() - t_start, 1)} s)")


if __name__ == "__main__":
    main()
