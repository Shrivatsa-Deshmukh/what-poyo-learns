"""Paired trial-bootstrap of ΔR² (perturbed − baseline) for one session.

Resampling unit = test interval (trial). The same resamples (fixed per session) are used for
every setting, so CIs are paired and comparable. R² is computed like
torchmetrics.functional.r2_score's default (per output dim, then uniform average) from
per-trial sufficient statistics, so 1000 resamples cost a few matrix products.
"""

import numpy as np

from src.perturb import stable_key


def r2_np(pred, target):
    ss_res = ((target - pred) ** 2).sum(0)
    ss_tot = ((target - target.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss_res / ss_tot))


class PairedTrialBootstrap:
    def __init__(self, session, timestamps, target, interval, pred_base, n_boot=1000):
        timestamps = np.asarray(timestamps, np.float64)
        trial = np.searchsorted(interval.start, timestamps, side="right") - 1
        assert np.all(trial >= 0) and np.all(timestamps <= interval.end[trial] + 1e-4), \
            "scored timestamp outside every test interval"
        n_trials = len(interval)
        self.timestamps = timestamps
        self.target = np.asarray(target, np.float64)
        self.trial, self.n_trials = trial, n_trials
        rng = np.random.default_rng([20261002, stable_key(session)])
        draws = rng.integers(0, n_trials, size=(n_boot, n_trials))
        self.W = np.stack([np.bincount(d, minlength=n_trials) for d in draws]).astype(float)
        t = self.target
        self.n = self.W @ np.bincount(trial, minlength=n_trials)   # (B,)
        st, st2 = self._per_trial(t), self._per_trial(t ** 2)        # (K, D)
        self.ss_tot = self.W @ st2 - (self.W @ st) ** 2 / self.n[:, None]  # (B, D)
        self.base_res = self._ss_res(pred_base)
        self.r2_base = r2_np(np.asarray(pred_base, np.float64), t)

    def _per_trial(self, x):
        return np.stack([np.bincount(self.trial, weights=x[:, d], minlength=self.n_trials)
                         for d in range(x.shape[1])], axis=1)

    def _ss_res(self, pred):
        return self.W @ self._per_trial((self.target - np.asarray(pred, np.float64)) ** 2)

    def delta(self, timestamps, pred):
        """Return (dR2, boot SE, 2.5%, 97.5%) for a perturbed prediction."""
        assert np.array_equal(np.asarray(timestamps, np.float64), self.timestamps), \
            "perturbed run scored different timestamps than baseline"
        d_boot = np.mean((self.base_res - self._ss_res(pred)) / self.ss_tot, axis=1)
        d = r2_np(np.asarray(pred, np.float64), self.target) - self.r2_base
        lo, hi = np.percentile(d_boot, [2.5, 97.5])
        return d, float(d_boot.std(ddof=1)), float(lo), float(hi)
