"""Cross-session transfer without training: evaluate POYO-MP with some of its learned unit or
session embeddings replaced by vectors taken from other sessions.

The replacement is done on the model's own embedding tables (`model.unit_emb.weight`,
`model.session_emb.weight`; one row per unit / session ID, row 0 = reserved padding word "NA"),
in memory, inside a context manager that restores the original rows bit-exactly on exit (also
on error). Unit and session IDs are unchanged, so tokenization, evaluation and stitching are
exactly those of the unmodified model; nothing is saved and no library code is touched.

Vector builders (pure numpy):
  electrode_mean_vectors  each target unit gets the mean embedding of the donor session's units
                          recorded on the same electrode (units on one electrode are not the same
                          neuron, so they are averaged); units without a donor unit on their
                          electrode get a fallback vector
  permuted_channels       the control: donor electrode labels permuted among themselves, so the
                          set of labels (and therefore the coverage) is unchanged but the
                          electrode correspondence is broken; with `groups` (e.g. brain area)
                          labels are only permuted within each group
  stat_matched_vectors    label-free matching: a target unit takes the donor unit whose firing
                          statistics (e.g. log rate, ISI CV; measured from spikes alone) are
                          nearest, either among the donor units on its own electrode or, when
                          the electrode recorded nothing on the donor day, among the donor units
                          of its brain area
"""

import contextlib

import numpy as np
import torch


def electrode_mean_vectors(target_channels, donor_channels, donor_vectors, fallback):
    """Returns (vectors (n_target, dim), matched (n_target,) bool)."""
    target_channels = np.asarray(target_channels)
    donor_channels = np.asarray(donor_channels)
    donor_vectors = np.asarray(donor_vectors, dtype=np.float64)
    assert len(donor_channels) == len(donor_vectors)
    means = {c: donor_vectors[donor_channels == c].mean(0) for c in np.unique(donor_channels)}
    fallback = np.asarray(fallback, dtype=np.float64)
    out = np.stack([means.get(c, fallback) for c in target_channels])
    matched = np.array([c in means for c in target_channels])
    return out, matched


def permuted_channels(donor_channels, rng, groups=None):
    """Donor electrode labels permuted among the donor's own labels (a random bijection).
    groups: optional label per unit (e.g. brain area); each electrode must lie in one group,
    and labels are then only permuted within their group."""
    donor_channels = np.asarray(donor_channels)
    if groups is None:
        groups = np.zeros(len(donor_channels), dtype=int)
    groups = np.asarray(groups)
    mapping = {}
    for g in np.unique(groups):
        labels = np.unique(donor_channels[groups == g])
        assert not set(labels) & set(mapping), "an electrode spans two groups"
        mapping.update(zip(labels, rng.permutation(labels)))
    return np.array([mapping[c] for c in donor_channels])



def stat_matched_vectors(target_channels, target_groups, target_feats, donor_channels,
                         donor_groups, donor_feats, donor_vectors, use_electrode=True):
    """Returns (vectors (n_target, dim), matched (n_target,) bool).

    feats: (n, k) standardised firing statistics. For each target unit:
      on an electrode the donor also recorded (if use_electrode): the donor unit on that electrode
        with the nearest statistics;
      otherwise: the donor unit of the same group (e.g. brain area) with the nearest statistics,
        or of the whole donor session if the group is absent there.
    matched marks the units that were matched by electrode."""
    tch, tg = np.asarray(target_channels), np.asarray(target_groups)
    dch, dg = np.asarray(donor_channels), np.asarray(donor_groups)
    tf = np.asarray(target_feats, dtype=np.float64)
    df = np.asarray(donor_feats, dtype=np.float64)
    dv = np.asarray(donor_vectors, dtype=np.float64)
    assert len(dch) == len(dg) == len(df) == len(dv) and len(tch) == len(tg) == len(tf)
    out = np.empty((len(tch), dv.shape[1]))
    matched = np.zeros(len(tch), bool)
    for i in range(len(tch)):
        cand = np.flatnonzero(dch == tch[i]) if use_electrode else np.array([], int)
        if len(cand):
            matched[i] = True
        else:
            cand = np.flatnonzero(dg == tg[i])
            if not len(cand):
                cand = np.arange(len(dg))
        dist = ((df[cand] - tf[i]) ** 2).sum(1)
        out[i] = dv[cand[np.argmin(dist)]]  # ties: first donor unit in table order
    return out, matched


@contextlib.contextmanager
def swapped_rows(model, unit_rows=None, session_rows=None):
    """Temporarily overwrite rows of the unit / session embedding tables.

    unit_rows, session_rows: {row index: vector (dim,)}. Row 0 (padding) may not be replaced.
    The original rows are restored bit-exactly when the block exits.
    """
    saved = []
    try:
        with torch.no_grad():
            for emb, rows in ((model.unit_emb, unit_rows), (model.session_emb, session_rows)):
                if not rows:
                    continue
                keys = sorted(rows)
                assert keys[0] > 0, "row 0 is the padding word and must not be replaced"
                w = emb.weight
                idx = torch.tensor(keys, dtype=torch.long, device=w.device)
                saved.append((w, idx, w[idx].clone()))
                new = np.stack([np.asarray(rows[k], dtype=np.float64) for k in keys])
                assert new.shape == (len(keys), w.shape[1]), "vector size does not match the table"
                w[idx] = torch.as_tensor(new, dtype=w.dtype, device=w.device)
        yield
    finally:
        with torch.no_grad():
            for w, idx, old in reversed(saved):
                w[idx] = old
