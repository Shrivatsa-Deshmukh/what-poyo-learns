"""Cross-day transfer analysis: cross-session transfer without training (CPU only, from saved results).

Inputs: results/transfer.csv (scripts/transfer/run_transfer.py), results/transfer_ridge.csv
(scripts/transfer/ridge_cross_day.py), results/sweep.csv (baseline R²).
"Kept" = R² of a condition ÷ the session's own baseline R² (POYO: unperturbed POYO-MP; ridge:
the session's own electrode decoder). Shuffle conditions are averaged over seeds within session.

Outputs:
  results/transfer_summary.csv    per condition (all / per monkey / per task): sessions, mean and
                                  median R², median kept [session-bootstrap 95% CI], share R² > 0
  results/transfer_tests.csv      paired Wilcoxon tests of the key contrasts, Holm-corrected
  results/transfer_gap.csv        POYO vs fixed ridge by days apart (median kept, share R² > 0)
Improved matching (run_transfer.py --set matching / combined; ridge_cross_day.py --pooled):
  results/transfer_improved_summary.csv  per condition (all / per monkey / per task), as above
  results/transfer_improved_tests.csv    paired Wilcoxon tests of the improved conditions
  results/transfer_improved_gap.csv      days apart: one donor day vs pooled days and rate matching,
                                         for POYO and the fixed ridge (same donors and pools)
The report's cross-day figures are drawn by scripts/docs/paper_figures.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

PROJECT = Path(__file__).resolve().parents[2]
R = PROJECT / "results"
GAPS = ["<=7d", "8-30d", "31-180d", ">180d"]
UNIT_CONDS = ["own_electrode_mean", "matched", "shuffled_within_area", "donor_mean"]
SESSION_CONDS = ["session_other_monkey"]


def load():
    d = pd.read_csv(R / "transfer.csv", float_precision="round_trip")
    sw = pd.read_csv(R / "sweep.csv", float_precision="round_trip")
    base = sw[sw.experiment == "baseline"].set_index("session").r2
    d["base"] = d.session.map(base)
    d["kept"] = d.r2 / d.base
    rid = pd.read_csv(R / "transfer_ridge.csv", float_precision="round_trip")
    rid["kept"] = rid.r2 / rid.r2_own_day
    return d, rid


def per_session(d, conds):
    """One row per (session, condition): seeds averaged."""
    x = d[d.condition.isin(conds)]
    return (x.groupby(["session", "monkey", "task", "condition"], as_index=False)
            .agg(r2=("r2", "mean"), kept=("kept", "mean"), coverage=("coverage", "mean"),
                 base=("base", "first")))


def boot_median(x, rng, n=1000):
    x = np.asarray(x, float)
    meds = np.median(rng.choice(x, (n, len(x)), replace=True), axis=1)
    return np.percentile(meds, [2.5, 97.5])


def summary(ses, rid):
    rng = np.random.default_rng(20261007)
    rows = []
    ridge = rid[rid.condition.isin(["cross_main", "own_day"])].assign(
        condition=lambda x: "ridge_" + x.condition)
    allx = pd.concat([ses, ridge[["session", "monkey", "task", "condition", "r2", "kept"]]])
    for cond, g in allx.groupby("condition"):
        for scope, gg in [("all", g)] + [(f"monkey {m}", x) for m, x in g.groupby("monkey")] \
                + [(f"task {t}", x) for t, x in g.groupby("task")]:
            lo, hi = boot_median(gg.kept, rng)
            rows.append(dict(condition=cond, scope=scope, n_sessions=len(gg),
                             r2_mean=gg.r2.mean(), r2_median=gg.r2.median(),
                             kept_median=gg.kept.median(), kept_ci_lo=lo, kept_ci_hi=hi,
                             share_r2_pos=float((gg.r2 > 0).mean())))
    return pd.DataFrame(rows)


def tests(ses, rid):
    w = ses.pivot_table(index="session", columns="condition", values="r2")
    k = ses.pivot_table(index="session", columns="condition", values="kept")
    base = ses.groupby("session").base.first()
    rk = rid[rid.condition == "cross_main"].set_index("session").kept
    contrasts = [
        ("matched vs shuffled within area (R²)", w.matched, w.shuffled_within_area),
        ("matched vs other day's average (R²)", w.matched, w.donor_mean),
        ("own-day electrode average vs baseline (R²)", w.own_electrode_mean, base),
        ("matched vs own-day electrode average (R²)", w.matched, w.own_electrode_mean),
        ("session swap, other monkey vs baseline (R²)", w.session_other_monkey, base),
        ("POYO matched vs fixed ridge, share of own R² kept", k.matched, rk),
    ]
    rows = []
    for name, a, b in contrasts:
        diff = (a - b).dropna()
        rows.append(dict(contrast=name, n_sessions=len(diff), mean_diff=diff.mean(),
                         median_diff=diff.median(), share_positive=float((diff > 0).mean()),
                         wilcoxon_p=wilcoxon(diff).pvalue))
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm"] = multipletests(out.wilcoxon_p, method="holm")[1]
    return out


def gap_table(d, rid):
    rows = []
    for gap in GAPS:
        p = d[(d.condition == "matched_gap") & (d.gap == gap)]
        s = d[(d.condition == "shuffled_gap") & (d.gap == gap)]
        r = rid[(rid.condition == "cross_gap") & (rid.gap == gap)]
        rows.append(dict(gap=gap, n_sessions=len(p), median_days=p.donor_days.median(),
                         median_coverage=p.coverage.median(),
                         poyo_kept_median=p.kept.median(), poyo_share_r2_pos=(p.r2 > 0).mean(),
                         poyo_shuffled_kept_median=s.kept.median(),
                         ridge_n=len(r), ridge_kept_median=r.kept.median(),
                         ridge_share_r2_pos=(r.r2 > 0).mean()))
    return pd.DataFrame(rows)


IMPROVED = ["matched", "matched_3days", "electrode_rate", "matched_3days_rate", "rate_only"]
GAP_IMPROVED = ["matched_gap", "electrode_rate_gap", "matched_3days_gap", "matched_3days_rate_gap"]


def load_improved(d):
    """Published rows plus the matching / combined runs and the pooled ridge, with checks."""
    base = d.groupby("session").base.first()
    m = pd.read_csv(R / "transfer_matching.csv", float_precision="round_trip")
    c = pd.read_csv(R / "transfer_combined.csv", float_precision="round_trip")
    pub = d[d.condition == "matched"].set_index("session").r2
    assert (m[m.condition == "matched"].set_index("session").r2 == pub).all(), "matched not reproduced"
    pub_gap = d[d.condition == "matched_gap"].set_index(["session", "gap"]).donor
    for cond in GAP_IMPROVED[1:]:
        x = c[c.condition == cond].set_index(["session", "gap"]).donor
        assert (x == pub_gap.loc[x.index]).all(), f"{cond}: donors differ from matched_gap"
    x = pd.concat([d[d.condition.isin(["matched", "matched_gap"])], m[m.condition != "matched"], c])
    x["base"] = x.session.map(base)
    x["kept"] = x.r2 / x.base
    rp = pd.read_csv(R / "transfer_ridge_pooled.csv", float_precision="round_trip")
    rp["kept"] = rp.r2 / rp.r2_own_day
    return x, rp


def improved_summary(x, rid, rp):
    rng = np.random.default_rng(20261008)
    ridge = pd.concat([rid[rid.condition == "cross_main"].assign(condition="ridge_cross_main"),
                       rp[rp.condition == "cross_3days"].assign(condition="ridge_cross_3days")])
    allx = pd.concat([x[x.condition.isin(IMPROVED)], ridge])
    rows = []
    for cond in IMPROVED + ["ridge_cross_main", "ridge_cross_3days"]:
        g = allx[allx.condition == cond]
        for scope, gg in [("all", g)] + [(f"monkey {k}", v) for k, v in g.groupby("monkey")] \
                + [(f"task {k}", v) for k, v in g.groupby("task")]:
            lo, hi = boot_median(gg.kept, rng)
            rows.append(dict(condition=cond, scope=scope, n_sessions=len(gg),
                             r2_median=gg.r2.median(), kept_median=gg.kept.median(),
                             kept_ci_lo=lo, kept_ci_hi=hi, share_r2_pos=float((gg.r2 > 0).mean()),
                             coverage_median=gg.coverage.median() if "coverage" in gg else np.nan))
    return pd.DataFrame(rows)


def improved_tests(x, rid, rp):
    k = x[x.gap.isna()].pivot_table(index="session", columns="condition", values="kept")
    r1 = rid[rid.condition == "cross_main"].set_index("session").kept
    r3 = rp[rp.condition == "cross_3days"].set_index("session").kept
    contrasts = [(f"{c} vs matched", k[c], k.matched) for c in IMPROVED[1:]] + [
        ("matched_3days_rate vs matched_3days", k.matched_3days_rate, k.matched_3days),
        ("matched_3days_rate vs electrode_rate", k.matched_3days_rate, k.electrode_rate),
        ("matched_3days vs electrode_rate", k.matched_3days, k.electrode_rate),
        ("ridge 3 days vs ridge 1 day", r3, r1),
        ("POYO matched_3days vs ridge 3 days", k.matched_3days, r3),
        ("POYO matched_3days_rate vs ridge 3 days", k.matched_3days_rate, r3),
    ]
    rows = []
    for name, a, b in contrasts:
        diff = (a - b).dropna()
        rows.append(dict(contrast=name + " (share of own R² kept)", n_sessions=len(diff),
                         median_diff=diff.median(), share_positive=float((diff > 0).mean()),
                         wilcoxon_p=wilcoxon(diff).pvalue))
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm"] = multipletests(out.wilcoxon_p, method="holm")[1]
    return out


def improved_gap(x, rid, rp):
    g = x[x.condition.isin(GAP_IMPROVED)]
    kept = g.pivot_table(index=["session", "gap"], columns="condition", values="kept")
    r2 = g.pivot_table(index=["session", "gap"], columns="condition", values="r2")
    cov = g.pivot_table(index=["session", "gap"], columns="condition", values="coverage")
    rows = []
    for gap in GAPS:
        k, r, cv = (t.xs(gap, level="gap") for t in (kept, r2, cov))
        r1 = rid[(rid.condition == "cross_gap") & (rid.gap == gap)]
        r3 = rp[(rp.condition == "cross_3days_gap") & (rp.gap == gap)]
        diff = k.matched_3days_rate_gap - k.matched_gap
        rows.append(dict(gap=gap, n_sessions=len(k),
                         **{f"{c}_kept_median": k[c].median() for c in GAP_IMPROVED},
                         best_share_r2_pos=float((r.matched_3days_rate_gap > 0).mean()),
                         coverage_1day=cv.matched_gap.median(), coverage_3days=cv.matched_3days_gap.median(),
                         best_minus_1day_median=diff.median(), best_better_share=float((diff > 0).mean()),
                         wilcoxon_p=wilcoxon(diff).pvalue,
                         ridge_1day_kept_median=r1.kept.median(), ridge_3days_kept_median=r3.kept.median(),
                         ridge_3days_share_r2_pos=float((r3.r2 > 0).mean())))
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm"] = multipletests(out.wilcoxon_p, method="holm")[1]
    return out


def main():
    d, rid = load()
    assert (d[d.condition == "own_swap"].r2 == d[d.condition == "own_swap"].base).all()
    ses = per_session(d, UNIT_CONDS + SESSION_CONDS)
    summ = summary(ses, rid)
    summ.to_csv(R / "transfer_summary.csv", index=False)
    t = tests(ses, rid)
    t.to_csv(R / "transfer_tests.csv", index=False)
    gap = gap_table(d, rid)
    gap.to_csv(R / "transfer_gap.csv", index=False)
    x, rp = load_improved(d)
    isum = improved_summary(x, rid, rp)
    isum.to_csv(R / "transfer_improved_summary.csv", index=False)
    itest = improved_tests(x, rid, rp)
    itest.to_csv(R / "transfer_improved_tests.csv", index=False)
    igap = improved_gap(x, rid, rp)
    igap.to_csv(R / "transfer_improved_gap.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ[summ.scope == "all"].round(3).to_string(index=False))
    print(t.round(4).to_string(index=False))
    print(gap.round(3).to_string(index=False))
    print(isum[isum.scope == "all"].round(3).to_string(index=False))
    print(itest.round(5).to_string(index=False))
    print(igap.round(3).T.to_string())


if __name__ == "__main__":
    main()
