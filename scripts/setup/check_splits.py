"""Compare the current (v0.2.0) and pre-c221ea9 splits for processed perich sessions.

Usage: python scripts/setup/check_splits.py <processed_dir>/perich_miller_population_2018/*.h5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torch_brain.data import Data  # noqa: E402

from src.splits import check_current_split_reproduced, make_splits, total_overlap  # noqa: E402


def dur(iv):
    return float((iv.end - iv.start).sum())


print(f"{'session':36s} {'cur_test':>9s} {'old_test':>9s} {'test_s':>7s} "
      f"{'in_old_train':>13s} {'in_old_test':>12s}")
for path in sys.argv[1:]:
    data = Data.load(path, lazy=False)
    check_current_split_reproduced(data)
    _, _, cur_test = make_splits(data, "current")
    old_train, _, old_test = make_splits(data, "pre_c221ea9")
    t = dur(cur_test)
    print(f"{data.session.id:36s} {len(cur_test):9d} {len(old_test):9d} {t:7.1f} "
          f"{total_overlap(cur_test, old_train) / t:13.1%} "
          f"{total_overlap(cur_test, old_test) / t:12.1%}")
print("current split reproduced exactly for all sessions")
