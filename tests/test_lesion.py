"""Invariant tests for src/lesion.py (silence vs remove)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.lesion import UnitLesion, lesion_order  # noqa: E402

S = "c_20131003_center_out_reaching"
H5 = PROJECT / f"data/processed/perich_miller_population_2018/{S}.h5"
CKPT = PROJECT / "checkpoints/poyo_mp.ckpt"
pytestmark = pytest.mark.skipif(not (H5.exists() and CKPT.exists()), reason="data/ckpt missing")


@pytest.fixture(scope="module")
def setup():
    from src.evaluate import DATA_ROOT, build_dataset, load_model, nested_id
    from torch_brain.data import Data
    from torch_brain.datasets import DatasetIndex
    model = load_model(CKPT, "cpu")
    _, nested = build_dataset(model, str(DATA_ROOT), [S])
    rid = nested_id(S)
    t0 = float(Data.load(H5, lazy=False).test_domain.start[3])

    def window(tf=None):
        nested.transform = tf
        try:
            return nested[DatasetIndex(rid, t0, t0 + 1.0)]
        finally:
            nested.transform = None
    ref = window()
    sid = str(ref.session.id)
    order = lesion_order(ref.units.id, sid, seed=0)
    return model, window, sid, order


def tok(model, w):
    t = model.tokenize(w)
    g = lambda k: np.asarray(t[k].obj if hasattr(t[k], "obj") else t[k])  # noqa: E731
    return g("input_unit_index"), g("input_timestamps"), g("input_token_type")


def spikes_by_id(w):
    ids = np.asarray([str(u) for u in w.units.id])
    return sorted(zip(ids[w.spikes.unit_index], np.round(w.spikes.timestamps, 9)))


@pytest.mark.parametrize("kind", ["remove", "silence"])
def test_fraction_zero_bit_identical(setup, kind):
    from torch_brain.batching import collate
    model, window, sid, _ = setup
    lesion = UnitLesion(kind, {sid: []})
    out_ts = torch.linspace(0, 0.99, 50)[None]
    a = model(**collate([model.tokenize(window())]), output_timestamps=out_ts)
    b = model(**collate([model.tokenize(window(lesion))]), output_timestamps=out_ts)
    assert torch.equal(a, b)


@pytest.mark.parametrize("n_les", [1, 10, 35])
def test_remove_no_tokens_of_lesioned_units(setup, n_les):
    model, window, sid, order = setup
    les = order[:n_les]
    w = window(UnitLesion("remove", {sid: les}))
    ui, ts, tt = tok(model, w)
    les_rows = set(model.unit_emb.tokenizer(les))
    assert not (set(ui.tolist()) & les_rows), "token of a lesioned unit reached the model"
    ref = window()
    assert (tt != 0).sum() == 2 * (len(ref.units.id) - n_les)          # delimiters of kept units only
    # kept units' spikes are untouched
    keep = [x for x in spikes_by_id(ref) if x[0] not in set(les)]
    assert spikes_by_id(w) == keep


@pytest.mark.parametrize("n_les", [1, 10, 35])
def test_silence_keeps_delimiters_drops_only_lesioned_spikes(setup, n_les):
    model, window, sid, order = setup
    les = order[:n_les]
    ref = window()
    w = window(UnitLesion("silence", {sid: les}))
    ui0, ts0, tt0 = tok(model, ref)
    ui1, ts1, tt1 = tok(model, w)
    d0, d1 = tt0 != 0, tt1 != 0
    assert np.array_equal(ui0[d0], ui1[d1]) and np.array_equal(ts0[d0], ts1[d1])  # delimiters same
    assert np.array_equal(tt0[d0], tt1[d1])
    assert spikes_by_id(w) == [x for x in spikes_by_id(ref) if x[0] not in set(les)]
    les_rows = set(model.unit_emb.tokenizer(les))
    assert not (set(ui1[~d1].tolist()) & les_rows), "spike token of a silenced unit remains"
    assert list(w.units.id) == list(ref.units.id)


def test_same_units_for_both_types_and_nested_sets(setup):
    _, window, sid, order = setup
    ref = window()
    assert order == lesion_order(ref.units.id, sid, seed=0)            # deterministic
    assert order != lesion_order(ref.units.id, sid, seed=1)            # seed matters
    assert set(order[:10]) <= set(order[:35])                          # nested across fractions


def test_unknown_unit_fails_loudly(setup):
    _, window, sid, _ = setup
    with pytest.raises(AssertionError):
        window(UnitLesion("remove", {sid: ["not_a_unit"]}))
