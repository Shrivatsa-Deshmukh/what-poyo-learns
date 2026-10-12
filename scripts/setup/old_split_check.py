"""Split check: POYO-MP baseline on the old (pre-c221ea9) test split + split verification.

For each session:
  1. R² on the old test split and on the current (v0.2.0) test split.
  2. Split check: R² on valid trials that lie inside vs outside each split version's
     training domain. If POYO-MP was trained on split version V, trials inside V's train
     domain should score clearly higher than trials in V's test set.

Usage: python scripts/setup/old_split_check.py <session> [<session> ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluate import (  # noqa: E402
    BRAINSET, DATA_ROOT, DEFAULT_CKPT, PROJECT, build_dataset, evaluate, load_model, nested_id,
    rotary_periods,
)
from src.splits import check_current_split_reproduced, make_splits  # noqa: E402
from torch_brain.data import Data  # noqa: E402

SEED = 0


def contained(trials, domain):
    """Boolean mask: trial interval lies entirely inside one interval of `domain`."""
    j = np.searchsorted(domain.start, trials.start, side="right") - 1
    ok = j >= 0
    jj = np.clip(j, 0, None)
    return ok & (domain.end[jj] >= trials.end) & (domain.start[jj] <= trials.start)


def trial_sets(data):
    """Interval sets to evaluate, keyed by name."""
    sets = {}
    valid = data.trials.select_by_mask(data.trials.is_valid)
    member = {}
    for version, tag in (("pre_c221ea9", "old"), ("current", "cur")):
        train, _, test = make_splits(data, version)
        sets[f"{tag}_test_split"] = test
        in_train = contained(valid, train)
        in_test = contained(valid, test)
        assert not np.any(in_train & in_test)
        member[tag] = (in_train, in_test)
        sets[f"{tag}:valid_trials_in_train"] = valid.select_by_mask(in_train)
        sets[f"{tag}:valid_trials_in_test"] = valid.select_by_mask(in_test)
    # trials on which the two split versions disagree (decisive for which one was used)
    sets["x:cur_test & old_train"] = valid.select_by_mask(member["cur"][1] & member["old"][0])
    sets["x:old_test & cur_train"] = valid.select_by_mask(member["old"][1] & member["cur"][0])
    sets["x:test in both"] = valid.select_by_mask(member["old"][1] & member["cur"][1])
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--root", default=str(DATA_ROOT))
    ap.add_argument("--out", default=str(PROJECT / "results" / "old_split_check.csv"))
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda"
    model = load_model(DEFAULT_CKPT, device)
    per = np.sort(rotary_periods(model))
    print(f"rotary: {len(per)} rotated freq pairs of head_dim/2="
          f"{model.rotary_emb.omega.numel()}, periods {per[0] * 1e3:.3f} ms .. {per[-1]:.3f} s")

    wrapped, _ = build_dataset(model, args.root, args.sessions)

    rows = []
    for s in args.sessions:
        data = Data.load(Path(args.root) / BRAINSET / f"{s}.h5", lazy=False)
        check_current_split_reproduced(data)
        task = "CO" if "center_out" in s else "RT"
        for name, iv in trial_sets(data).items():
            dur = iv.end - iv.start
            short = dur < model.sequence_length
            if short.any():
                print(f"  {s} {name}: {short.sum()}/{len(iv)} intervals < 1 s "
                      f"(shortest {dur.min():.2f} s) -> excluded (sampler cannot window them)")
            iv = iv.select_by_mask(~short)
            res = evaluate(model, wrapped, {nested_id(s): iv}, device=device)[nested_id(s)]
            rows.append(dict(session=s, task=task, set=name, n_intervals=len(iv),
                             n_short_excluded=int(short.sum()), n_windows=res["n_windows"],
                             n_scored=res["n_scored"], r2=res["r2"]))
            print(f"{s:36s} {name:28s} n={len(iv):3d} win={res['n_windows']:4d} "
                  f"scored={res['n_scored']:5d} R2={res['r2']:.4f}")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
