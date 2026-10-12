"""Lesion study analysis: remove vs silence, and POYO vs a ridge decoder refitted on the kept units.

Inputs: results/lesion.csv (GPU lesion runs), results/lesion_ridge.csv, results/sweep.csv
Outputs: results/lesion_summary.csv, results/lesion_vs_ridge.csv. The report's missing-neuron
figures are drawn by scripts/docs/paper_figures.py.
"""

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

PROJECT = Path(__file__).resolve().parents[2]
R = PROJECT / "results"
ORDER = ["frac10", "frac25", "frac50", "frac75", "keep30", "keep20"]


def load():
    b = pd.read_csv(R / "lesion.csv", float_precision="round_trip")
    base = pd.read_csv(R / "sweep.csv", float_precision="round_trip")
    base = base[base.experiment == "baseline"].set_index("session").r2
    b["r2_base"] = b.session.map(base)
    b["rel_loss"] = -b.dr2 / b.r2_base
    return b


def remove_vs_silence(b):
    w = b.pivot_table(index=["session", "task", "condition", "seed", "in_training", "n_units"],
                      columns="lesion_type", values="dr2").reset_index()
    w["diff"] = w["remove"] - w["silence"]  # > 0: removing hurts less than silencing
    ses = w.groupby(["session", "task", "condition", "in_training", "n_units"]).agg(
        remove=("remove", "mean"), silence=("silence", "mean"), diff=("diff", "mean"),
        diff_seed_sd=("diff", "std"), remove_seed_sd=("remove", "std")).reset_index()
    rows = []
    for c in ORDER:
        for task in ("all", "CO", "RT"):
            g = ses[(ses.condition == c) & ((ses.task == task) if task != "all" else True)]
            if len(g) < 5:
                continue
            rows.append(dict(condition=c, task=task, n_sessions=len(g),
                             frac_in_training=g.in_training.mean(),
                             dr2_remove_mean=g.remove.mean(), dr2_remove_sem=g.remove.sem(),
                             dr2_silence_mean=g.silence.mean(), dr2_silence_sem=g.silence.sem(),
                             diff_mean=g["diff"].mean(), diff_sem=g["diff"].sem(),
                             frac_sessions_remove_better=(g["diff"] > 0).mean(),
                             seed_sd_of_diff=g.diff_seed_sd.mean(),
                             seed_sd_of_remove=g.remove_seed_sd.mean(),
                             wilcoxon_p=wilcoxon(g["diff"]).pvalue))
    s = pd.DataFrame(rows)
    for task, g in s.groupby("task"):
        s.loc[g.index, "wilcoxon_p_holm"] = multipletests(g.wilcoxon_p, method="holm")[1]
    return s, ses


def vs_ridge(b):
    rr = pd.read_csv(R / "lesion_ridge.csv")
    base = rr[rr.condition == "none"].set_index(["session", "variant"]).r2_test
    rr = rr[rr.condition != "none"].copy()
    rr["ridge_rel_loss"] = [(base[(s, v)] - r) / base[(s, v)]
                            for s, v, r in zip(rr.session, rr.variant, rr.r2_test)]
    pr = b[b.lesion_type == "remove"][["session", "condition", "seed", "rel_loss", "task"]]
    m = rr.merge(pr, on=["session", "condition", "seed"])
    ses = m.groupby(["variant", "condition", "session", "task"])[["ridge_rel_loss", "rel_loss"]].mean().reset_index()
    rows = []
    for v in ("grid",):
        for c in ORDER:
            g = ses[(ses.variant == v) & (ses.condition == c)]
            dd = g.rel_loss - g.ridge_rel_loss
            rows.append(dict(variant=v, condition=c, n_sessions=len(g),
                             poyo_remove_rel_loss=g.rel_loss.mean(),
                             ridge_refit_rel_loss=g.ridge_rel_loss.mean(),
                             diff_poyo_minus_ridge=dd.mean(), diff_sem=dd.sem(),
                             wilcoxon_p=wilcoxon(dd).pvalue))
    out = pd.DataFrame(rows)
    for v, g in out.groupby("variant"):
        out.loc[g.index, "wilcoxon_p_holm"] = multipletests(g.wilcoxon_p, method="holm")[1]
    return out, ses


def main():
    b = load()
    summ, _ = remove_vs_silence(b)
    summ.to_csv(R / "lesion_summary.csv", index=False)
    vr, _ = vs_ridge(b)
    vr.to_csv(R / "lesion_vs_ridge.csv", index=False)
    pd.set_option("display.width", 240)
    print("=== remove vs silence (per-session ΔR², seeds averaged)")
    print(summ.round(4).to_string(index=False))
    print("\n=== POYO remove vs ridge refitted\n", vr.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
