"""Tests for src/codebook.py (neuron letters) and the held-out logic of
scripts/codebook/run_codebook.py (no model needed: random embedding tables with the real unit
table)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts" / "codebook"))

from src.codebook import (  # noqa: E402
    bits, blocks, fit_centroids, fit_rotation, group_means, kmeans_quantize, nearest,
    pca_truncate, product_quantize, residual_quantize, session_folds,
)

UNITS = PROJECT / "results" / "emb_units.csv"
needs_units = pytest.mark.skipif(not UNITS.exists(), reason="results/emb_units.csv missing")


def clusters(n_per=30, k=4, dim=8, seed=0, spread=0.01):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, dim)) * 5
    X = np.concatenate([c + spread * rng.normal(size=(n_per, dim)) for c in centres])
    return X, centres


# ------------------------------------------------------------------- one alphabet
def test_k1_gives_the_training_mean():
    rng = np.random.default_rng(0)
    Xtr, Xte = rng.normal(size=(50, 6)), rng.normal(size=(7, 6))
    v, c = kmeans_quantize(Xtr, Xte, 1)
    assert np.allclose(v, Xtr.mean(0)) and (c == 0).all()


def test_k_equal_to_n_gives_the_vectors_back_exactly():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 5))
    v, _ = kmeans_quantize(X, X, len(X))
    assert np.allclose(v, X)


def test_well_separated_clusters_are_recovered_by_their_letter():
    X, centres = clusters()
    v, codes = kmeans_quantize(X, centres, 4)
    assert len(set(codes)) == 4 and np.allclose(v, centres, atol=0.01)


def test_kmeans_is_deterministic_for_a_seed():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 8))
    assert np.array_equal(fit_centroids(X, 16, 3), fit_centroids(X, 16, 3))


def test_nearest_matches_brute_force():
    rng = np.random.default_rng(3)
    X, C = rng.normal(size=(40, 5)), rng.normal(size=(9, 5))
    brute = np.array([np.argmin(((C - x) ** 2).sum(1)) for x in X])
    assert np.array_equal(nearest(X, C), brute)


def test_too_few_training_vectors_is_an_error():
    with pytest.raises(AssertionError):
        fit_centroids(np.zeros((3, 2)), 4)


# ------------------------------------------------------------------- parallel letters
def test_rotation_is_orthonormal_and_ordered_by_variance():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 6)) * np.array([5, 4, 3, 2, 1, 0.5])
    mu, vt = fit_rotation(X)
    assert np.allclose(vt @ vt.T, np.eye(6))
    var = ((X - mu) @ vt.T).var(0)
    assert np.all(np.diff(var) <= 1e-9)


def test_blocks_are_round_robin_and_cover_every_dimension_once():
    b = blocks(8, 4)
    assert [x.tolist() for x in b] == [[0, 4], [1, 5], [2, 6], [3, 7]]
    with pytest.raises(AssertionError):
        blocks(10, 4)


def test_product_with_one_block_equals_kmeans_in_the_rotated_space():
    rng = np.random.default_rng(5)
    Xtr, Xte = rng.normal(size=(150, 6)), rng.normal(size=(10, 6))
    v_pq, c_pq = product_quantize(Xtr, Xte, 1, 8, seed=0)
    mu, vt = fit_rotation(Xtr)
    v_k, c_k = kmeans_quantize((Xtr - mu) @ vt.T, (Xte - mu) @ vt.T, 8, seed=0)
    assert np.array_equal(c_pq[:, 0], c_k) and np.allclose(v_pq, v_k @ vt + mu)


def test_product_reproduces_vectors_when_letters_suffice_and_makes_more_words_than_letters():
    rng = np.random.default_rng(6)
    Z = np.repeat(rng.normal(size=(9, 4)), 10, axis=0)          # 9 distinct vectors
    v, codes = product_quantize(Z, Z, 2, 9, seed=0)
    assert np.allclose(v, Z, atol=1e-8) and codes.shape == (len(Z), 2)
    X = rng.normal(size=(500, 8))
    _, c = product_quantize(X, X, 2, 8, seed=0)                  # 8 + 8 letters
    assert len({tuple(r) for r in c}) > 16                       # more words than letters


# ------------------------------------------------------------------- coarse-to-fine letters
def test_residual_stage_one_equals_kmeans():
    rng = np.random.default_rng(7)
    Xtr, Xte = rng.normal(size=(120, 5)), rng.normal(size=(9, 5))
    v_r, c_r = residual_quantize(Xtr, Xte, [8], seed=0)
    v_k, c_k = kmeans_quantize(Xtr, Xte, 8, seed=0)
    assert np.array_equal(c_r[:, 0], c_k) and np.allclose(v_r, v_k)


def test_residual_stages_never_increase_training_error():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(400, 8))
    errs = [((X - residual_quantize(X, X, [16] * n, seed=0)[0]) ** 2).sum() for n in (1, 2, 3)]
    assert errs[0] >= errs[1] >= errs[2]


def test_residual_prefix_codes_are_stable():
    # stage j uses seed + j, so the first stages of a longer code equal the shorter code
    rng = np.random.default_rng(9)
    Xtr, Xte = rng.normal(size=(200, 6)), rng.normal(size=(15, 6))
    _, c2 = residual_quantize(Xtr, Xte, [8, 8], seed=0)
    _, c3 = residual_quantize(Xtr, Xte, [8, 8, 8], seed=0)
    assert np.array_equal(c2, c3[:, :2])


# ------------------------------------------------------------------- controls, bits
def test_pca_full_rank_is_identity_and_d0_like_rank_drop():
    rng = np.random.default_rng(10)
    Xtr, Xte = rng.normal(size=(100, 5)), rng.normal(size=(8, 5))
    assert np.allclose(pca_truncate(Xtr, Xte, 5), Xte)
    e = [((Xtr - pca_truncate(Xtr, Xtr, d)) ** 2).sum() for d in (1, 2, 3, 4)]
    assert all(a >= b for a, b in zip(e, e[1:]))


def test_group_means():
    Xtr = np.array([[0.0, 0], [2, 0], [0, 4]])
    v, c = group_means(Xtr, ["M1", "M1", "PMd"], ["PMd", "M1"])
    assert np.allclose(v, [[0, 4], [1, 0]]) and c.tolist() == [1, 0]
    with pytest.raises(AssertionError):
        group_means(Xtr, ["M1", "M1", "PMd"], ["X"])


def test_bits():
    assert bits("kmeans", 256) == 8
    assert bits("product", 16, 4) == 16
    assert bits("residual", 16, 3) == 12


# ------------------------------------------------------------------- folds, held-out logic
def test_session_folds_cover_every_monkey_in_every_fold():
    sessions = [f"{m}_{i}" for m, n in (("c", 20), ("m", 10), ("j", 3)) for i in range(n)]
    monkeys = [s[0] for s in sessions]
    f = session_folds(sessions, monkeys, 5, seed=0)
    assert set(f) == set(sessions) and set(f.values()) == set(range(5))
    for m in ("c", "m"):
        assert {f[s] for s in sessions if s[0] == m} == set(range(5))
    assert f == session_folds(sessions, monkeys, 5, seed=0)
    sizes = np.bincount(list(f.values()))
    assert sizes.max() - sizes.min() <= 1


@needs_units
@pytest.mark.parametrize("scheme", ["sessions", "monkeys"])
def test_codebooks_never_see_the_held_out_session(scheme):
    import run_codebook as rc
    units = pd.read_csv(UNITS)
    grp = rc.groups_for(scheme, units)
    assert set(grp) == set(units.session)
    if scheme == "monkeys":
        meta = units.drop_duplicates("session").set_index("session")
        assert all(grp[s] == f"monkey_{meta.loc[s, 'monkey']}" for s in grp)
    else:
        assert len(set(grp.values())) == 5
    # a held-out unit planted far from everything must not be its own letter
    rng = np.random.default_rng(0)
    W = rng.normal(size=(units.emb_row.max() + 1, 8))
    g = grp[units.session.iloc[0]]
    in_g = units.session.map(grp) == g
    test = units[in_g]
    W[test.emb_row.iloc[0]] = 1e3
    res = rc.code_group(units[~in_g], test, W)
    v, _ = res["kmeans_K1024"]
    assert np.abs(v).max() < 100                     # no centroid was pulled to the outlier
    assert np.allclose(res["own"][0], W[test.emb_row.to_numpy()])


@needs_units
def test_random_letters_permute_within_each_session():
    import run_codebook as rc
    units = pd.read_csv(UNITS)
    grp = rc.groups_for("sessions", units)
    rng = np.random.default_rng(1)
    W = rng.normal(size=(units.emb_row.max() + 1, 8))
    in_g = units.session.map(grp) == "fold0"
    test = units[in_g]
    res = rc.code_group(units[~in_g], test, W)
    for k in rc.RANDOM_K:
        v, c = res[f"kmeans_K{k}"]
        vr, cr = res[f"random_K{k}"]
        for s in test.session.unique():
            m = (test.session == s).to_numpy()
            assert sorted(c[m]) == sorted(cr[m])          # same letters, other units
        assert (c != cr).mean() > 0.5


def test_condition_specs_are_unique():
    import run_codebook as rc
    names = [c for c, _, _ in rc.condition_specs()]
    assert len(names) == len(set(names))


# ------------------------------------------------------------------- cached batches (model)
S = "c_20131003_center_out_reaching"
H5 = PROJECT / f"data/processed/perich_miller_population_2018/{S}.h5"
CKPT = PROJECT / "checkpoints/poyo_mp.ckpt"
needs_model = pytest.mark.skipif(not (H5.exists() and CKPT.exists()), reason="data/ckpt missing")


@pytest.fixture(scope="module")
def cached():
    import torch
    from src.evaluate import (DATA_ROOT, build_dataset, iter_sessions, load_model,
                              session_batches)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(CKPT, device)
    wrapped, nested = build_dataset(model, str(DATA_ROOT), [S])
    _, ref = next(iter_sessions(model, wrapped, nested, [S], None, device=device))
    batches, interval = session_batches(model, wrapped, nested, S, device=device)
    return model, batches, interval, ref, device


@needs_model
def test_cached_batches_in_full_precision_equal_the_normal_evaluation(cached):
    from src.evaluate import evaluate_batches
    model, batches, interval, ref, device = cached
    r = evaluate_batches(model, batches, S, interval, device)
    assert r["r2"] == ref["r2"] and np.array_equal(r["pred"], ref["pred"])
    r2 = evaluate_batches(model, batches, S, interval, device)      # batches are reusable
    assert r2["r2"] == ref["r2"]


@needs_model
def test_swapped_rows_act_on_cached_batches_and_are_restored(cached):
    from src.evaluate import evaluate_batches
    from src.transfer import swapped_rows
    model, batches, interval, ref, device = cached
    units = pd.read_csv(UNITS)
    rows = units[units.session == S].emb_row.to_numpy()
    with swapped_rows(model, {r: np.zeros(model.unit_emb.weight.shape[1]) for r in rows}):
        r0 = evaluate_batches(model, batches, S, interval, device)
    assert r0["r2"] < ref["r2"] - 0.1
    assert evaluate_batches(model, batches, S, interval, device)["r2"] == ref["r2"]


@needs_model
def test_fp16_is_close_to_full_precision(cached):
    import torch
    from src.evaluate import evaluate_batches
    model, batches, interval, ref, device = cached
    if device != "cuda" or not torch.cuda.is_available():
        pytest.skip("fp16 autocast path is for CUDA")
    r = evaluate_batches(model, batches, S, interval, device, amp=True)
    assert abs(r["r2"] - ref["r2"]) < 1e-3


# ------------------------------------------------------------------- direction-only letters
def test_on_direction_keeps_each_vectors_own_length_and_codes_direction_only():
    from src.codebook import on_direction
    rng = np.random.default_rng(11)
    Xtr = rng.normal(size=(200, 6))
    Xte = rng.normal(size=(12, 6)) * rng.uniform(0.5, 5, size=(12, 1))
    v, codes = on_direction(kmeans_quantize, Xtr, Xte, 8, 0)
    assert np.allclose(np.linalg.norm(v, axis=1), np.linalg.norm(Xte, axis=1))
    # scaling a training or test vector does not change its letter
    v2, codes2 = on_direction(kmeans_quantize, Xtr * 3.0, Xte * 7.0, 8, 0)
    assert np.array_equal(codes, codes2) and np.allclose(v2, v * 7.0)
    # one direction per test vector: exact
    v3, _ = on_direction(kmeans_quantize, Xte, Xte, len(Xte), 0)
    assert np.allclose(v3, Xte)


def test_unit_rows_rejects_zero_vectors():
    from src.codebook import unit_rows
    with pytest.raises(AssertionError):
        unit_rows(np.zeros((2, 3)))


@needs_units
def test_direction_set_conditions_match_the_specs():
    import run_codebook as rc
    units = pd.read_csv(UNITS)
    grp = rc.groups_for("sessions", units)
    rng = np.random.default_rng(2)
    W = rng.normal(size=(units.emb_row.max() + 1, 8))
    in_g = units.session.map(grp) == "fold1"
    res = rc.code_group(units[~in_g], units[in_g], W, cset="direction")
    names = [c for c, _, _ in rc.condition_specs("direction")]
    assert list(res) == names and len(names) == len(set(names))
    x = W[units[in_g].emb_row.to_numpy()]
    v, _ = res["mean_length"]
    lengths = np.linalg.norm(v, axis=1)
    assert np.allclose(lengths, lengths[0])                        # one common length
    assert np.allclose(v / lengths[:, None], x / np.linalg.norm(x, axis=1, keepdims=True))
