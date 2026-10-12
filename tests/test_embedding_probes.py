"""Tests for src/embedding_probes.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import embedding_probes as ep  # noqa: E402


def toy(n_sess=20, per=40, d=32, seed=0, signal=0.0, session_offset=0.0):
    rng = np.random.default_rng(seed)
    sessions = np.repeat([f"s{i:02d}" for i in range(n_sess)], per)
    X = rng.normal(size=(len(sessions), d))
    X += session_offset * rng.normal(size=(n_sess, d))[np.repeat(np.arange(n_sess), per)]
    w = rng.normal(size=d)
    y = signal * (X @ w) / np.linalg.norm(w) + rng.normal(size=len(sessions))
    folds = np.repeat(np.arange(n_sess) % 5, per)
    return X, y, sessions, folds, rng


def test_no_session_in_train_and_test():
    _, _, sessions, folds, _ = toy()
    ep.check_folds(sessions, folds)
    bad = folds.copy()
    bad[0] = (bad[0] + 1) % 5  # one unit of session s00 moved to another fold
    with pytest.raises(AssertionError):
        ep.check_folds(sessions, bad)


def test_standardisation_and_penalty_use_training_data_only():
    X, y, sessions, folds, _ = toy(signal=1.0)
    tr = folds != 0
    m = ep.fit_ridge(X[tr], y[tr], sessions[tr])
    np.testing.assert_allclose(m.named_steps["scale"].mean_, X[tr].mean(0))
    # corrupting the held-out rows must not change the fitted model
    X2 = X.copy()
    X2[~tr] *= 1e6
    m2 = ep.fit_ridge(X2[tr], y[tr], sessions[tr])
    assert m.named_steps["ridge"].alpha == m2.named_steps["ridge"].alpha
    np.testing.assert_allclose(m.named_steps["ridge"].coef_, m2.named_steps["ridge"].coef_)


def test_random_embeddings_give_chance():
    X, y, sessions, folds, rng = toy(signal=0.0)
    r = ep.cv_continuous(X, y, sessions, folds, rank=True)
    assert ep.r2(r["y"], r["pred"]) < 0.02
    null = ep.null_continuous(X, y, sessions, folds, r["alpha"], 50, rng)
    assert np.percentile(null, 95) < 0.03


def test_planted_signal_recovered_and_beats_null():
    X, y, sessions, folds, rng = toy(signal=2.0)
    r = ep.cv_continuous(X, y, sessions, folds, rank=False)
    score = ep.r2(r["y"], r["pred"])
    assert score > 0.6  # signal variance 4 vs noise 1 -> ceiling 0.8
    null = ep.null_continuous(X, y, sessions, folds, r["alpha"], 30, rng, rank=False)
    assert score > null.max()


def test_rank_transform_uses_training_ecdf_only():
    tr = np.array([1.0, 2.0, 3.0, 4.0])
    a = ep.ecdf_normal_scores(tr, np.array([2.5, 100.0]))
    b = ep.ecdf_normal_scores(tr, np.array([2.5, -100.0]))
    assert a[0] == b[0]                    # a held-out value never shifts another's score
    assert a[1] == ep.ecdf_normal_scores(tr, np.array([4.0]))[0]  # clipped to training range


def test_direction_probe_planted():
    rng = np.random.default_rng(3)
    n_sess, per, d = 20, 40, 16
    sessions = np.repeat([f"s{i}" for i in range(n_sess)], per)
    theta = rng.uniform(-np.pi, np.pi, len(sessions))
    X = np.column_stack([np.cos(theta), np.sin(theta), rng.normal(size=(len(theta), d - 2))])
    X[:, :2] += 0.3 * rng.normal(size=(len(theta), 2))
    folds = np.array([int(s[1:]) % 5 for s in sessions])
    r = ep.cv_direction(X, theta, sessions, folds)
    assert ep.circ_corr(theta, r["pred"]) > 0.8
    assert np.median(ep.ang_err(theta, r["pred"])) < 0.35


def test_session_mean_loo_and_within_session():
    y = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    s = np.array(["a", "a", "a", "b", "b"])
    np.testing.assert_allclose(ep.session_mean_loo(y, s), [2.5, 2.0, 1.5, 20.0, 10.0])
    # a prediction that only knows the session mean has within-session R² <= 0
    pred = np.array([2.0, 2.0, 2.0, 15.0, 15.0])
    assert ep.within_session_r2(y, pred, s) <= 1e-12


def test_variance_explained_by_session():
    X, _, sessions, _, _ = toy(session_offset=3.0)
    _, overall = ep.variance_explained(X, sessions)
    assert overall > 0.8
    X0, _, s0, _, _ = toy(session_offset=0.0)
    assert ep.variance_explained(X0, s0)[1] < 0.1


def test_session_circmean_loo():
    th = np.array([0.0, np.pi / 2, np.pi / 2, 1.0])
    s = np.array(["a", "a", "a", "b"])
    out = ep.session_circmean_loo(th, s)
    np.testing.assert_allclose(out[0], np.pi / 2)            # mean of the other two
    np.testing.assert_allclose(out[1], np.pi / 4)            # mean of 0 and π/2
    assert np.isnan(out[3])                                  # alone in its session
