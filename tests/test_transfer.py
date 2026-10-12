"""Tests for src/transfer.py (cross-session transfer without training)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.transfer import (  # noqa: E402
    electrode_mean_vectors, permuted_channels, stat_matched_vectors, swapped_rows,
)

S = "c_20131003_center_out_reaching"
H5 = PROJECT / f"data/processed/perich_miller_population_2018/{S}.h5"
CKPT = PROJECT / "checkpoints/poyo_mp.ckpt"
needs_model = pytest.mark.skipif(not (H5.exists() and CKPT.exists()), reason="data/ckpt missing")


# ------------------------------------------------------------------- vector builders
def test_electrode_mean_averages_units_on_the_same_electrode():
    donor_ch = np.array(["a", "a", "b", "c"])
    donor_v = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [5.0, 5.0]])
    fallback = np.array([-1.0, -1.0])
    v, m = electrode_mean_vectors(["a", "b", "z", "a"], donor_ch, donor_v, fallback)
    assert np.allclose(v, [[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0], [2.0, 0.0]])
    assert m.tolist() == [True, True, False, True]


def test_electrode_mean_of_a_session_with_itself_is_identity_for_single_unit_electrodes():
    rng = np.random.default_rng(0)
    ch = np.array([f"e{i}" for i in range(20)])
    vec = rng.normal(size=(20, 8))
    v, m = electrode_mean_vectors(ch, ch, vec, np.zeros(8))
    assert m.all() and np.allclose(v, vec)


def test_permuted_channels_is_a_bijection_on_the_donor_labels():
    rng = np.random.default_rng(1)
    ch = np.array(["a", "a", "b", "c", "d", "d", "e"])
    p = permuted_channels(ch, rng)
    assert sorted(set(p)) == sorted(set(ch))                       # same labels -> same coverage
    for c in set(ch):  # units of one electrode move together
        assert len(set(p[ch == c])) == 1
    assert np.array_equal(permuted_channels(ch, np.random.default_rng(1)), p)    # seeded


def test_permuted_channels_within_groups_stays_inside_each_group():
    rng = np.random.default_rng(3)
    ch = np.array(["m1a", "m1b", "m1c", "m1c", "pa", "pb", "pc"])
    area = np.array(["M1", "M1", "M1", "M1", "PMd", "PMd", "PMd"])
    for _ in range(20):
        p = permuted_channels(ch, rng, area)
        assert set(p[area == "M1"]) == {"m1a", "m1b", "m1c"}
        assert set(p[area == "PMd"]) == {"pa", "pb", "pc"}
        assert len(set(p[ch == "m1c"])) == 1
    with pytest.raises(AssertionError):                    # an electrode in two groups
        permuted_channels(np.array(["x", "x"]), rng, np.array(["M1", "PMd"]))


def test_permuted_channels_keeps_coverage_of_any_target():
    rng = np.random.default_rng(2)
    donor = np.array([f"e{i}" for i in range(0, 60, 2)] * 2)
    target = np.array([f"e{i}" for i in range(0, 90, 3)])
    _, m1 = electrode_mean_vectors(target, donor, np.ones((len(donor), 4)), np.zeros(4))
    _, m2 = electrode_mean_vectors(target, permuted_channels(donor, rng), np.ones((len(donor), 4)),
                                   np.zeros(4))
    assert np.array_equal(m1, m2)


# --------------------------------------------------------------------- swapped_rows
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unit_emb = torch.nn.Embedding(6, 3)
        self.session_emb = torch.nn.Embedding(4, 3)


def _donor():
    ch = np.array(["e1", "e1", "e2", "e9"])
    grp = np.array(["M1", "M1", "M1", "PMd"])
    feats = np.array([[0.0, 0.0], [2.0, 0.0], [5.0, 5.0], [1.0, 1.0]])
    vecs = np.arange(8.0).reshape(4, 2)
    return ch, grp, feats, vecs


def test_stat_matched_picks_the_nearest_unit_on_the_same_electrode():
    ch, grp, feats, vecs = _donor()
    v, m = stat_matched_vectors(["e1"], ["M1"], [[1.9, 0.1]], ch, grp, feats, vecs)
    assert m.tolist() == [True] and np.array_equal(v[0], vecs[1])


def test_stat_matched_falls_back_to_the_nearest_unit_of_the_same_area():
    ch, grp, feats, vecs = _donor()
    # electrode e7 is empty on the donor day; nearest M1 unit by statistics is the e2 unit
    v, m = stat_matched_vectors(["e7"], ["M1"], [[4.0, 4.0]], ch, grp, feats, vecs)
    assert m.tolist() == [False] and np.array_equal(v[0], vecs[2])
    # a PMd unit only considers PMd donor units, however close the M1 ones are
    v, _ = stat_matched_vectors(["e7"], ["PMd"], [[0.0, 0.0]], ch, grp, feats, vecs)
    assert np.array_equal(v[0], vecs[3])


def test_stat_matched_without_electrodes_ignores_them():
    ch, grp, feats, vecs = _donor()
    v, m = stat_matched_vectors(["e1"], ["M1"], [[5.0, 5.0]], ch, grp, feats, vecs,
                                use_electrode=False)
    assert m.tolist() == [False] and np.array_equal(v[0], vecs[2])


def test_swapped_rows_overwrites_and_restores_exactly():
    m = Tiny()
    u0, s0 = m.unit_emb.weight.detach().clone(), m.session_emb.weight.detach().clone()
    with swapped_rows(m, {2: np.ones(3), 5: np.full(3, 7.0)}, {1: np.zeros(3)}):
        assert torch.equal(m.unit_emb.weight[2], torch.ones(3))
        assert torch.equal(m.unit_emb.weight[5], torch.full((3,), 7.0))
        assert torch.equal(m.session_emb.weight[1], torch.zeros(3))
        assert torch.equal(m.unit_emb.weight[[0, 1, 3, 4]], u0[[0, 1, 3, 4]])  # others untouched
    assert torch.equal(m.unit_emb.weight, u0) and torch.equal(m.session_emb.weight, s0)


def test_swapped_rows_restores_on_error():
    m = Tiny()
    u0 = m.unit_emb.weight.detach().clone()
    with pytest.raises(RuntimeError):
        with swapped_rows(m, {3: np.ones(3)}):
            raise RuntimeError("boom")
    assert torch.equal(m.unit_emb.weight, u0)


def test_swapped_rows_refuses_padding_row_and_wrong_size():
    m = Tiny()
    u0 = m.unit_emb.weight.detach().clone()
    with pytest.raises(AssertionError):
        with swapped_rows(m, {0: np.ones(3)}):
            pass
    with pytest.raises(AssertionError):
        with swapped_rows(m, {1: np.ones(4)}):
            pass
    assert torch.equal(m.unit_emb.weight, u0)


# ------------------------------------------------------------------- real model
@pytest.fixture(scope="module")
def setup():
    from src.evaluate import DATA_ROOT, build_dataset, load_model, nested_id
    from torch_brain.batching import collate
    from torch_brain.data import Data
    from torch_brain.datasets import DatasetIndex
    model = load_model(CKPT, "cpu")
    _, nested = build_dataset(model, str(DATA_ROOT), [S])
    t0 = float(Data.load(H5, lazy=False).test_domain.start[3])
    win = nested[DatasetIndex(nested_id(S), t0, t0 + 1.0)]
    out_ts = torch.linspace(0, 0.99, 50)[None]

    def run():
        with torch.no_grad():
            return model(**collate([model.tokenize(win)]), output_timestamps=out_ts)
    unit_rows = model.unit_emb.tokenizer([str(u) for u in win.units.id])
    session_row = model.session_emb.tokenizer(str(win.session.id))
    return model, run, unit_rows, session_row


@needs_model
def test_own_vectors_reproduce_the_model_bit_exactly(setup):
    model, run, unit_rows, session_row = setup
    ref = run()
    W, Ws = model.unit_emb.weight.detach().numpy(), model.session_emb.weight.detach().numpy()
    with swapped_rows(model, {r: W[r].copy() for r in unit_rows},
                      {session_row: Ws[session_row].copy()}):
        same = run()
    assert torch.equal(ref, same)


@needs_model
def test_swapping_changes_the_output_and_is_undone(setup):
    model, run, unit_rows, session_row = setup
    ref = run()
    w0 = model.unit_emb.weight.detach().clone()
    mean = model.unit_emb.weight.detach().numpy()[1:].mean(0)
    with swapped_rows(model, {r: mean for r in unit_rows}):
        changed = run()
    assert not torch.allclose(ref, changed)
    assert torch.equal(model.unit_emb.weight, w0)
    assert torch.equal(run(), ref)


@needs_model
def test_session_swap_changes_the_output(setup):
    model, run, _, session_row = setup
    ref = run()
    other = 1 if session_row != 1 else 2
    other_vec = model.session_emb.weight[other].detach().numpy()
    with swapped_rows(model, session_rows={session_row: other_vec}):
        changed = run()
    assert not torch.allclose(ref, changed)
    assert torch.equal(run(), ref)
