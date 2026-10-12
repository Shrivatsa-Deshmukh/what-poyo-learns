"""Lesion study: silence vs remove lesions of POYO-MP input units (test-time only).

Applied to each sliced window after slicing and before POYO.tokenize (same place as
src/perturb.py; set as the NestedDataset transform).

  remove   the unit is gone: its spikes are dropped AND it is removed from data.units, so the
           tokenizer emits no [START]/[END] tokens for it (POYO.tokenize builds delimiters from
           data.units.id, models/poyo.py:271-280). spikes.unit_index is re-indexed to the kept
           units exactly as UnitDropout(reset_index=True) does
           (transforms/unit_dropout.py:143-162) — the operation used in POYO-MP training.
  silence  the unit is present but never fires: only its spikes are dropped; data.units and
           therefore its delimiter tokens are unchanged.

The lesioned set is fixed per session and seed (same units in every window, same units for
both lesion types): a seeded permutation of the session's units; a condition lesions its first
n_lesioned units, so sets are nested across conditions of the same seed.
"""

import numpy as np

from src.perturb import stable_key
from torch_brain.data import Data


def lesion_order(unit_ids, session, seed):
    """Seeded permutation of a session's unit IDs (order in which units get lesioned)."""
    rng = np.random.default_rng([20261004, stable_key(session), seed])
    return [str(u) for u in np.asarray(unit_ids)[rng.permutation(len(unit_ids))]]


class UnitLesion:
    """lesioned: {session id (as in data.session.id): set of unit IDs (as in data.units.id)}."""

    def __init__(self, kind, lesioned):
        assert kind in ("remove", "silence"), kind
        self.kind = kind
        self.lesioned = {s: set(map(str, u)) for s, u in lesioned.items()}

    def __call__(self, data: Data) -> Data:
        les = self.lesioned.get(str(data.session.id), set())
        if not les:
            return data  # fraction 0: identity (same object)
        ids = np.asarray([str(u) for u in data.units.id])
        hit = np.isin(ids, list(les))
        assert hit.sum() == len(les), "lesioned unit not found in this session"
        sp = data.spikes
        keep_spike = ~hit[sp.unit_index]
        data.spikes = sp.select_by_mask(keep_spike)
        if self.kind == "remove":
            relabel = np.full(len(ids), -1)
            relabel[~hit] = np.arange((~hit).sum())
            data.units = data.units.select_by_mask(~hit)
            data.spikes.unit_index = relabel[data.spikes.unit_index]
            assert (data.spikes.unit_index >= 0).all()
        return data

    def __repr__(self):
        return f"UnitLesion({self.kind}, {sum(map(len, self.lesioned.values()))} units)"
