"""Embedding study, first check: verify the unit-embedding matrix, its vocabulary and unit metadata.

Checks (assertions): embedding rows = 9,725 units + 1 reserved 'NA' row; every unit of the 99
POYO-MP sessions maps to exactly one row and every non-'NA' row maps to one of those units;
unit IDs parse into area (M1/PMd) and electrode; the electrode-table index is mapped to the NWB
physical channel label (consistent within each monkey, checked). Writes results/emb_units.csv (one row per unit
with session, monkey, task, area, electrode, embedding row) and prints a summary.
"""

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluate import (  # noqa: E402
    DATA_ROOT, DEFAULT_CKPT, PROJECT, TASK, TRAIN_RECORDING_IDS, build_dataset, load_model,
    nested_id,
)

ID_RE = re.compile(r"^perich_miller_population_2018/(?P<session>[cmjt]_\d{8}_[a-z_]+)/"
                   r"group_electrode_group_(?P<area>[A-Za-z0-9]+)/elec(?P<elec>\d+)/"
                   r"(?P<kind>unit|multiunit)_(?P<num>\d+)$")


def main():
    model = load_model(DEFAULT_CKPT, "cpu")
    W = model.unit_emb.weight.detach().numpy()
    vocab = model.unit_emb.vocab
    print(f"unit_emb: {type(model.unit_emb).__name__}, weight {W.shape}, vocab {len(vocab)} words, "
          f"padding_idx {model.unit_emb.padding_idx}, row0 word {next(iter(vocab))!r}, "
          f"|row0| = {np.linalg.norm(W[0]):.4f}")
    print(f"session_emb: weight {tuple(model.session_emb.weight.shape)}; "
          f"token_type_emb {tuple(model.token_type_emb.weight.shape)}")
    assert W.shape[0] == len(vocab) and next(iter(vocab)) == "NA" and vocab["NA"] == 0

    _, nested = build_dataset(model, str(DATA_ROOT), TRAIN_RECORDING_IDS)
    rows = []
    for s in sorted(TRAIN_RECORDING_IDS):
        data = nested.get_recording(nested_id(s))
        ids = [str(u) for u in data.units.id]
        for local, uid in enumerate(ids):
            m = ID_RE.match(uid)
            assert m, uid
            assert m["session"] == s
            rows.append(dict(unit_id=uid, session=s, monkey=s[0].upper(), task=TASK(s),
                             area=m["area"], electrode=int(m["elec"]), kind=m["kind"],
                             unit_number=int(m["num"]), local_index=local,
                             emb_row=vocab[uid]))
    df = pd.DataFrame(rows)
    assert df.unit_id.is_unique and df.emb_row.is_unique
    covered = set(df.emb_row)
    assert covered == set(range(1, len(vocab))), "rows not covered by the 99 sessions' units"
    df.to_csv(PROJECT / "results/emb_units.csv", index=False)

    print(f"\n{len(df)} units in 99 sessions; each maps to exactly one row; rows 1..{len(vocab) - 1} "
          f"all covered; 'NA' (row 0) excluded")
    print("unit kind:", dict(Counter(df.kind)))
    print("area x monkey:\n", pd.crosstab(df.area, df.monkey).to_string())
    print("sessions with both M1 and PMd:", int((df.groupby("session").area.nunique() > 1).sum()))
    print("units per session: min/median/max",
          df.groupby("session").size().agg(["min", "median", "max"]).to_dict())
    print("electrodes: min/max", df.electrode.min(), df.electrode.max(),
          "| units per (session, electrode) max:", df.groupby(["session", "electrode"]).size().max())
    for s in ("c_20131003_center_out_reaching", "c_20131009_random_target_reaching"):
        g = df[df.session == s]
        print(f"\n{s}: {len(g)} units; area {dict(Counter(g.area))}; electrodes {g.electrode.nunique()}")
        print(g.head(3)[["unit_id", "area", "electrode", "unit_number", "emb_row"]].to_string(index=False))
    data = nested.get_recording(nested_id("c_20131003_center_out_reaching"))
    print("\nunits ArrayDict fields:", data.units.keys())


def add_channel_labels():
    """Map each unit's electrode-table index to the NWB physical channel label
    (e.g. 'elecM1bankApin1'); the index alone is a per-file table row
    (torch_brain/utils/dandi.py:100-101). Adds column 'channel' to results/emb_units.csv."""
    import glob

    from pynwb import NWBHDF5IO
    df = pd.read_csv(PROJECT / "results/emb_units.csv")
    raw = PROJECT / "data/raw/perich_miller_population_2018"
    labels, consistent = {}, {}
    for s in sorted(df.session.unique()):
        sub = s[0].upper()
        task = "CO" if "center_out" in s else "RT"
        f = glob.glob(str(raw / f"sub-{sub}/sub-{sub}_ses-{task}-{s[2:10]}_behavior+ecephys.nwb"))
        assert len(f) == 1, (s, f)
        with NWBHDF5IO(f[0], "r") as io:
            e = io.read().electrodes.to_dataframe()
        lab = dict(zip(e.index.astype(int), e["label"].astype(str)))
        g = e["group_name"].str.replace("electrode_group_", "")
        labels[s] = lab
        consistent.setdefault(sub, {})
        for i, (l, a) in enumerate(zip(e["label"], g)):
            consistent[sub].setdefault(int(e.index[i]), set()).add(l)
    df["channel"] = [labels[s][e] for s, e in zip(df.session, df.electrode)]
    assert all(a in c for a, c in zip(df.area, df.channel)), "area/channel label mismatch"
    df.to_csv(PROJECT / "results/emb_units.csv", index=False)
    for sub, d in consistent.items():
        n_bad = sum(len(v) > 1 for v in d.values())
        print(f"monkey {sub}: electrode index -> label consistent across sessions: "
              f"{'yes' if n_bad == 0 else f'NO ({n_bad} indices map to >1 label)'}")
    pairs = df.groupby(["monkey", "channel"]).session.nunique()
    print("channels recorded in >1 session:", int((pairs > 1).sum()), "of", len(pairs))


if __name__ == "__main__":
    main()
    add_channel_labels()
