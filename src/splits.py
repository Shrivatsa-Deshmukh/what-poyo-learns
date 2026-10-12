"""Reconstruct the perich_miller_population_2018 train/valid/test splits.

Two split versions exist:
  - "current": torch_brain 0.2.0 pipeline (= brainsets >= c221ea9, 2025-03-19) splits
    the *valid* trials only. This is what is stored in the processed .h5 files.
  - "pre_c221ea9": the brainsets pipeline before commit c221ea9 split *all* trials.
    The public POYO-MP checkpoint (uploaded 2025-03-20) was expected to be trained on it, but
    scripts/setup/old_split_check.py showed it was trained on the current split.

All other trial/phase extraction logic is identical between the two versions, and the
processed file stores the full trial table (`data.trials`) that was passed to the split.
So the old split can be regenerated from the current processed file.

`split_trials` below is a verbatim copy of
torch_brain/pipeline/brainsets-pipelines/perich_miller_population_2018/pipeline.py:365-382
(identical to brainsets@c221ea9^ prepare_data.py:181-198). It is copied rather than
imported because that module imports `dandi` at top level, which is not in f_models.
`check_current_split_reproduced` proves the copy matches the stored split.
"""

import numpy as np
from sklearn.model_selection import train_test_split


def split_trials(trials, test_size=0.2, valid_size=0.1, random_state=42):
    num_trials = len(trials)
    train_size = 1.0 - test_size - valid_size

    train_valid_ids, test_ids = train_test_split(
        np.arange(num_trials), test_size=test_size, random_state=random_state
    )
    train_ids, valid_ids = train_test_split(
        train_valid_ids,
        test_size=valid_size / (train_size + valid_size),
        random_state=random_state,
    )

    train_trials = trials.select_by_mask(np.isin(np.arange(num_trials), train_ids))
    valid_trials = trials.select_by_mask(np.isin(np.arange(num_trials), valid_ids))
    test_trials = trials.select_by_mask(np.isin(np.arange(num_trials), test_ids))

    return train_trials, valid_trials, test_trials


def make_splits(data, version):
    """Return (train_domain, valid_domain, test_domain) for a split version."""
    if version == "current":
        trials = data.trials.select_by_mask(data.trials.is_valid)
    elif version == "pre_c221ea9":
        trials = data.trials
    else:
        raise ValueError(f"unknown split version {version!r}")
    _, valid, test = split_trials(trials, test_size=0.2, valid_size=0.1, random_state=42)
    # same as pipeline.py:199-201
    train = data.domain.difference((valid | test).dilate(1.0))
    return train, valid, test


def _same_intervals(a, b):
    return (len(a) == len(b) and np.array_equal(a.start, b.start)
            and np.array_equal(a.end, b.end))


def check_current_split_reproduced(data):
    train, valid, test = make_splits(data, "current")
    assert _same_intervals(train, data.train_domain), "train split not reproduced"
    assert _same_intervals(valid, data.valid_domain), "valid split not reproduced"
    assert _same_intervals(test, data.test_domain), "test split not reproduced"


def total_overlap(a, b):
    """Total duration (s) of the intersection of two Interval sets."""
    inter = a & b
    return float((inter.end - inter.start).sum()) if len(inter) else 0.0
