"""Evaluate the pretrained POYO-MP checkpoint on perich_miller_population_2018.

Reuses the evaluation path of torch_brain v0.2.0 examples/poyo/train_simple.py:122-146
(validation loop) unchanged in substance:
  - SequentialFixedWindowSampler, window = model.sequence_length (1 s), step = window / 2
  - PoyoDatasetWrapper(dataset, model.tokenize) builds inputs and targets
  - per-session BehaviorStitcher averages predictions that share a timestamp
  - torchmetrics.functional.r2_score on timestamps inside readout_config.eval_interval
    (CO: movement_phases.reach_period, RT: movement_phases.random_period)
Only the sampling intervals differ: we pass the split we want (old or current test).
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torchmetrics

PROJECT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = PROJECT / "external" / "torch_brain_v0.2.0" / "examples" / "poyo"
sys.path.insert(0, str(EXAMPLE_DIR))

from datasets.poyo_mp import TRAIN_RECORDING_IDS, PoyoMPDataset  # noqa: E402
from datasets.wrapper import PoyoDatasetWrapper  # noqa: E402
from utils import BehaviorStitcher, move_to_device  # noqa: E402

from torch_brain.batching import collate  # noqa: E402
from torch_brain.datasets import NestedDataset  # noqa: E402
from torch_brain.datasets.PerichMillerPopulation2018 import (  # noqa: E402
    PerichMillerPopulation2018,
)
from torch_brain.models import POYO  # noqa: E402
from torch_brain.samplers import SequentialFixedWindowSampler  # noqa: E402
from torch_brain.utils.stitcher import stitch  # noqa: E402

BRAINSET = "perich_miller_population_2018"
DEFAULT_CKPT = PROJECT / "checkpoints" / "poyo_mp.ckpt"
DATA_ROOT = PROJECT / "data" / "processed"  # brainsets processed dir (holds BRAINSET/)


class SubsetPoyoMPDataset(PoyoMPDataset):
    """PoyoMPDataset (example, unmodified readout config) restricted to some sessions.

    PoyoMPDataset.__init__ hard-codes all 99 TRAIN_RECORDING_IDS; we skip it and call the
    parent constructor with a subset so that only prepared sessions are opened.
    """

    def __init__(self, root, recording_ids, transform=None):
        missing = sorted(set(recording_ids) - set(TRAIN_RECORDING_IDS))
        assert not missing, f"not POYO-MP sessions: {missing}"
        PerichMillerPopulation2018.__init__(
            self, root, recording_ids=recording_ids, transform=transform
        )


def load_model(ckpt_path=DEFAULT_CKPT, device="cuda"):
    model = POYO.load_pretrained(checkpoint_path=str(ckpt_path))
    return model.to(device).eval()


def build_dataset(model, root, recording_ids):
    """Nested so ids get the 'perich_miller_population_2018/' namespace the checkpoint
    vocab uses. Returns (wrapped_dataset, nested_dataset)."""
    nested = NestedDataset({BRAINSET: SubsetPoyoMPDataset(root, recording_ids)})
    # every unit / session must be in the checkpoint vocab (tokenizer raises KeyError too)
    for rid in nested.recording_ids:
        data = nested.get_recording(rid)
        assert data.session.id in model.session_emb.vocab, data.session.id
        unknown = [u for u in data.units.id if u not in model.unit_emb.vocab]
        assert not unknown, f"{rid}: {len(unknown)} units not in checkpoint vocab"
    return PoyoDatasetWrapper(nested, model.tokenize), nested


@torch.no_grad()
def evaluate(model, wrapped_ds, sampling_intervals, batch_size=32, device="cuda"):
    """sampling_intervals: {nested_recording_id: Interval}. Returns dict of
    {nested_recording_id: dict(r2, n_scored, n_windows, pred, target)}."""
    sampler = SequentialFixedWindowSampler(
        sampling_intervals=sampling_intervals,
        window_length=model.sequence_length,
        step=model.sequence_length / 2.0,
    )
    loader = torch.utils.data.DataLoader(
        wrapped_ds, sampler=sampler, batch_size=batch_size, num_workers=0,
        collate_fn=collate,
    )
    return _score(model, loader, list(sampling_intervals), device)


@torch.no_grad()
def _score(model, batches, rids, device="cuda"):
    """Forward pass, stitching and R² for an iterable of (X, Y) batches (see evaluate)."""
    model.eval()
    stitchers = {rid: BehaviorStitcher() for rid in rids}
    n_windows = {rid: 0 for rid in rids}
    for X, Y in batches:
        X, Y = move_to_device((X, Y), device)
        # .float(): no-op in full precision; under fp16 autocast, stitch and score in fp32
        pred = model(**X, output_timestamps=Y["timestamps"]).float()
        for i in range(len(pred)):
            _mask = Y["eval_mask"][i]
            _rid = Y["session_id"][i]
            n_windows[_rid] += 1
            # stitched on CPU: identical result to GPU stitching (checked), but CUDA
            # index_add_ in torch_brain.utils.stitcher.stitch is non-deterministic at ~1e-7
            stitchers[_rid].update(
                preds=pred[i][_mask].cpu(),
                targets=Y["values"][i][_mask].cpu(),
                timestamps=(Y["timestamps"][i][_mask] + Y["absolute_start"][i]).cpu(),
            )
    out = {}
    for rid, stitcher in stitchers.items():
        assert stitcher.pred_cache, f"{rid}: no windows evaluated"
        pred, target = stitcher.compute()
        assert len(pred) > 1, f"{rid}: no scored timestamps"
        # same stitch call as BehaviorStitcher.compute (examples/poyo/utils.py:217-223),
        # to also keep the stitched timestamps (needed to map points to trials)
        ts, _ = stitch(torch.concat(stitcher.timestamps_cache),
                       torch.concat(stitcher.target_cache))
        assert len(ts) == len(pred)
        out[rid] = dict(
            r2=torchmetrics.functional.r2_score(pred, target).item(),
            n_scored=len(pred), n_windows=n_windows[rid],
            pred=pred.cpu().numpy(), target=target.cpu().numpy(),
            timestamps=ts.cpu().numpy(),
        )
    return out


def session_batches(model, wrapped, nested, session, batch_size=32, device="cuda"):
    """The tokenized test windows of one session, built once and moved to the device, plus the
    test Interval. Tokens hold unit / session indices, not vectors, so the batches stay valid
    when embedding rows are swapped; re-scoring them skips the data loading."""
    rid = nested_id(session)
    nested.transform = None
    interval = nested.get_sampling_intervals("test")[rid]
    sampler = SequentialFixedWindowSampler(
        sampling_intervals={rid: interval},
        window_length=model.sequence_length,
        step=model.sequence_length / 2.0,
    )
    loader = torch.utils.data.DataLoader(
        wrapped, sampler=sampler, batch_size=batch_size, num_workers=0, collate_fn=collate,
    )
    return [move_to_device(b, device) for b in loader], interval


def evaluate_batches(model, batches, session, interval, device="cuda", amp=False):
    """As iter_sessions for one session, on batches from session_batches. amp=True runs the
    forward pass under fp16 autocast (about 2.8x faster on the RTX 4050; R² within ~4e-5)."""
    import time

    rid = nested_id(session)
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        r = _score(model, batches, [rid], device)[rid]
    if device == "cuda":
        torch.cuda.synchronize()
    r["runtime_s"] = time.perf_counter() - t0
    r["interval"] = interval
    return r


def nested_id(session):
    return f"{BRAINSET}/{session}"


def rotary_periods(model):
    omega = model.rotary_emb.omega
    omega = omega[omega > 0]
    return (2 * np.pi / omega).cpu().numpy()


TASK = lambda s: "CO" if "center_out" in s else "RT"  # noqa: E731


def iter_sessions(model, wrapped, nested, sessions, perturbation=None, device="cuda"):
    """Yield (session, result) for the test split of each session (one evaluate() call per
    session), with an optional spike-timing perturbation inserted after slicing and before
    tokenization. result has r2, n_windows, n_scored, runtime_s, pred, target, timestamps,
    interval (the session's test Interval)."""
    import time

    nested.transform = perturbation  # NestedDataset.__getitem__ applies it post-slice
    intervals = nested.get_sampling_intervals("test")
    try:
        for s in sessions:
            rid = nested_id(s)
            t0 = time.perf_counter()
            r = evaluate(model, wrapped, {rid: intervals[rid]}, device=device)[rid]
            if device == "cuda":
                torch.cuda.synchronize()
            r["runtime_s"] = time.perf_counter() - t0
            r["interval"] = intervals[rid]
            yield s, r
    finally:
        nested.transform = None
