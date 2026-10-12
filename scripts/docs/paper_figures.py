"""Figures of the paper (main.tex) drawn from the result files, at print size (scripts/docs/figstyle.py).

  fig2_results            Figure 2 (main): one panel per finding, all 99 sessions
  figS1_transfer          Figure S1: another day's embeddings (schematic, conditions, days apart)
  figS2_unit_embeddings   Figure S2: what the embeddings encode (tuning, probes, direction, electrode)
  figS3_neuron_letters    Figure S3: neuron letters (codebook size, other monkeys, continuous
                          control, direction-only letters)
  figS4_missing_neurons   Figure S4: missing neurons (schematic, remove vs silence, refitted ridge)
  figS5_spike_timing      Figure S5: spike timing (schematic, small bins, CO / RT curves)

Each is saved as figures/<name>.pdf and .png. Reads only files in results/ (and, for the
example tuning curve, one recording), so it never changes a result. Schematic panels are drawn by
scripts/docs/make_diagrams.py. Colours: figstyle.py.
Usage: python scripts/docs/paper_figures.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (  # noqa: E402
    BLUE, BLUE_LIGHT, FS_SMALL, GREY, GRID, INK, INK2, LIGHT, ORANGE, PROJECT, axes, canvas, figure,
    letter, log_ticks, save, style,
)
import make_diagrams as md  # noqa: E402

sys.path.insert(0, str(PROJECT))
from src import unit_properties as up  # noqa: E402

R = PROJECT / "results"


def read(name):
    return pd.read_csv(R / name, float_precision="round_trip")


def kept_by_session():
    """Share of own R² kept, per condition × session (shuffle seeds averaged), and per gap row."""
    d = pd.concat([read("transfer.csv"), read("transfer_matching.csv").query("condition != 'matched'"),
                   read("transfer_combined.csv")])
    sw = read("sweep.csv")
    d["kept"] = d.r2 / d.session.map(sw[sw.experiment == "baseline"].set_index("session").r2)
    return d


# ----------------------------------------------------------------------------- shared panels
CONDS = [("frac10", "10% lost"), ("frac25", "25% lost"), ("frac50", "50% lost"),
         ("frac75", "75% lost"), ("keep30", "keep 30"), ("keep20", "keep 20")]
COND_Y = np.arange(len(CONDS))[::-1] + np.array([0.4] * 4 + [0] * 2)


def dumbbell(ax, y, a, b, ca, cb, la, lb, mb="o", ea=None, eb=None):
    for yy, xa, xb in zip(y, a, b):
        ax.plot([xa, xb], [yy, yy], color=LIGHT, lw=1.6, zorder=1, solid_capstyle="round")
    # the second series first, so the main condition is drawn on top where they overlap
    for x, e, c, m, lab, z in ((b, eb, cb, mb, lb, 3), (a, ea, ca, "o", la, 4)):
        if e is not None:
            ax.errorbar(x, y, xerr=e, fmt="none", ecolor=c, elinewidth=0.8, capsize=0, zorder=z)
        ax.scatter(x, y, s=16, color=c, marker=m, edgecolors="white", linewidths=0.6, zorder=z,
                   label=lab)
    h, lab = ax.get_legend_handles_labels()
    return h[::-1], lab[::-1]


def panel_remove_silence(ax, title):
    ls = read("lesion_summary.csv")
    ls = ls[ls.task == "all"].set_index("condition").loc[[c for c, _ in CONDS]]
    h, lab = dumbbell(ax, COND_Y, ls.dr2_remove_mean, ls.dr2_silence_mean, BLUE, ORANGE,
                      "remove (as in training)", "silence (start / end kept)",
                      ea=ls.dr2_remove_sem, eb=ls.dr2_silence_sem)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(COND_Y, [lab for _, lab in CONDS])
    ax.set_xlim(-0.62, 0.02)
    ax.set_xlabel("ΔR² vs all units (mean ± SEM)")
    ax.legend(h, lab, frameon=False, loc="upper left")
    style(ax, title, grid="x")


def panel_pref_dir(ax, title):
    pr = pd.read_csv(R / "emb_probe_preds.csv.gz")
    pr = pr[pr.target == "pref_dir"]
    ax.scatter(np.degrees(pr.y), np.degrees(pr.pred), s=0.6, color=BLUE, alpha=0.25, lw=0,
               rasterized=True)
    ax.plot([-180, 180], [-180, 180], color=INK2, lw=0.7, ls="--")
    for f in (ax.set_xlim, ax.set_ylim):
        f(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_aspect("equal")
    ax.set_xlabel("true preferred direction (deg)")
    ax.set_ylabel("predicted from embedding (deg)")
    style(ax, title.format(n=len(pr)), grid="both")


def panel_days(ax, title):
    gap, imp = read("transfer_gap.csv"), read("transfer_improved_gap.csv")
    assert (gap.gap == imp.gap).all()
    x = np.arange(len(gap))
    ax.plot(x, imp.matched_3days_rate_gap_kept_median, "o-", color=BLUE, lw=1.5, ms=4,
            mec="white", mew=0.5, label="POYO, 3 days + rate")
    ax.plot(x, gap.poyo_kept_median, "o-", color=BLUE_LIGHT, lw=1.0, ms=3.5, mfc="white",
            label="POYO, 1 day")
    ax.plot(x, imp.ridge_3days_kept_median, "s--", color=GREY, lw=1.3, ms=3.5, mec="white",
            mew=0.5, label="ridge, 3 days")
    ax.plot(x, gap.ridge_kept_median, "s--", color=LIGHT, lw=1.0, ms=3, mfc="white",
            label="ridge, 1 day")
    ax.plot(x, gap.poyo_shuffled_kept_median, "o:", color=LIGHT, lw=1.2, ms=3, mec="white",
            mew=0.5, label="POYO, electrodes shuffled")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axhline(1, color=INK2, lw=0.6, ls="--")
    labels = ["≤ 7", "8–30", "31–180", "> 180"]
    ax.set_xticks(x, [f"{lab} d\n{100 * c:.0f}→{100 * c3:.0f}%"
                      for lab, c, c3 in zip(labels, imp.coverage_1day, imp.coverage_3days)])
    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(-1.15, 1.1)
    ax.set_xlabel("days apart (units on a matched electrode: 1 → 3 days)")
    ax.set_ylabel("share of own R² kept (median)")
    ax.legend(frameon=False, loc="lower left", ncol=1)
    style(ax, title)


def poyo_relative_loss():
    """Per-session relative loss (−ΔR² ÷ own baseline R², seeds averaged) of POYO under bin
    shuffle at every bin size, as in scripts/timing/ridge_control_summary.py (which stops at
    10 ms)."""
    sw = read("sweep.csv")
    base = sw[sw.experiment == "baseline"].set_index("session").r2
    g = sw[sw.experiment == "bin_shuffle"]
    g = g.groupby(["session", "task", "bin_ms"], as_index=False).dr2.mean()
    g["frac_loss"] = -g.dr2 / g.session.map(base)
    return g


def latent_line(ax, y, ha="right"):
    ax.axvline(125, color=INK2, lw=0.6, ls=":")
    ax.text(118 if ha == "right" else 132, y, "125 ms latent\nspacing", color=INK2,
            fontsize=FS_SMALL, ha=ha, va="top")


# ------------------------------------------------------------------------------- Figure 2
def fig_results():
    sys.path.insert(0, str(PROJECT / "scripts" / "timing"))
    from ridge_control_summary import per_session_curves

    poyo = poyo_relative_loss()
    ridge = per_session_curves()
    ridge = ridge[ridge.decoder == "ridge (grid)"]
    pub = read("binned_ridge_summary.csv")
    pub = pub[pub.decoder == "POYO bin shuffle"].set_index(["task", "bin_ms"]).frac_loss
    chk = poyo.groupby(["task", "bin_ms"]).frac_loss.mean().loc[pub.index]
    assert np.allclose(chk, pub, atol=1e-12)

    H = 5.42
    fig = figure(H)
    panel_days(axes(fig, 0.55, 3.2, 2.6, 1.9), "1 · Another day's embeddings, no learning")
    panel_pref_dir(axes(fig, 4.3, 3.2, 1.9, 1.9), "2 · Embeddings encode preferred direction")
    panel_remove_silence(axes(fig, 0.62, 0.38, 2.53, 1.95), "3 · Missing neurons: absent beats silent")
    ax = axes(fig, 3.98, 0.38, 2.65, 1.95)
    for g, col, ls, mk, lab in ((poyo, BLUE, "-", "o", "POYO, bin shuffle"),
                                (ridge, GREY, "--", "s", "ridge, trained per bin size")):
        m = g.groupby("bin_ms").frac_loss.agg(["mean", "sem"])
        ax.errorbar(m.index, 100 * m["mean"], yerr=100 * m["sem"], color=col, ls=ls, lw=1.3,
                    marker=mk, ms=3.5, mec="white", mew=0.5, capsize=0, label=lab)
    latent_line(ax, 30)
    ax.axhline(0, color=INK2, lw=0.6)
    log_ticks(ax, [1, 2, 5, 10, 20, 50, 100, 200, 500])
    ax.set_xlabel("bin size (ms)")
    ax.set_ylabel("loss (% of the decoder's own R²)")
    ax.legend(frameon=False, loc="upper left")
    style(ax, "4 · Fine spike timing is barely used")
    for x, y, L in ((0.0, 5.4, "A"), (3.62, 5.4, "B"), (0.0, 2.62, "C"), (3.5, 2.62, "D")):
        letter(fig, x, y, L)
    save(fig, "fig2_results")


# ------------------------------------------------------------------------------ Figure S1
UNIT_CONDS = [("own_electrode_mean", "own day (averaged)", INK2),
              ("matched_3days_rate", "3 days + rate", BLUE),
              ("matched_3days", "3 days", BLUE),
              ("electrode_rate", "1 day + rate", BLUE),
              ("matched", "1 day (matched)", BLUE),
              ("rate_only", "rate only, any electrode", LIGHT),
              ("shuffled_within_area", "electrodes shuffled", LIGHT),
              ("donor_mean", "other day's average", LIGHT)]
GAP_VARIANTS = [("matched_3days_rate_gap", "3 days + rate", BLUE, "o", "-", 1.5),
                ("matched_3days_gap", "3 days", BLUE, "s", "--", 1.0),
                ("electrode_rate_gap", "1 day + rate", BLUE_LIGHT, "^", "--", 1.0),
                ("matched_gap", "1 day (matched)", BLUE_LIGHT, "o", "-", 1.0)]


def boot_median_ci(x, rng, n=1000):
    x = np.asarray(x, float)
    return np.percentile(np.median(rng.choice(x, (n, len(x))), axis=1), [2.5, 97.5])


def fig_transfer():
    d = kept_by_session()
    per = d[d.gap.isna()].groupby(["condition", "session"]).kept.mean()
    H = 5.15
    fig = figure(H)
    md.transfer_panel(canvas(fig, 0.02, 3.5, 6.73, 1.45), 6.73, 1.45)

    ax = axes(fig, 1.32, 0.42, 2.0, 2.75)
    rng = np.random.default_rng(0)
    y0 = np.arange(len(UNIT_CONDS))[::-1]
    for yy, (c, lab, col) in zip(y0, UNIT_CONDS):
        v = per.loc[c].to_numpy()
        ax.scatter(v, yy + rng.uniform(-0.2, 0.2, len(v)), s=3, color=col, alpha=0.8, lw=0,
                   rasterized=True)
        ax.plot([np.median(v)] * 2, [yy - 0.32, yy + 0.32], color=INK, lw=1.4)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.axvline(1, color=INK2, lw=0.6, ls="--")
    ax.set_yticks(y0, [lab for _, lab, _ in UNIT_CONDS])
    ax.set_xlim(-0.6, 1.1)
    ax.set_xticks([-0.5, 0, 0.5, 1])
    ax.set_xlabel("share of own R² kept (bar: median)")
    style(ax, "Unit embeddings replaced, 99 sessions", grid="x")

    ax = axes(fig, 3.98, 0.55, 2.65, 2.62)
    imp = read("transfer_improved_gap.csv").set_index("gap")
    gaps = ["<=7d", "8-30d", "31-180d", ">180d"]
    rng = np.random.default_rng(20261009)
    for k, (cond, lab, col, mk, ls, lw) in enumerate(GAP_VARIANTS):
        g = d[d.condition == cond]
        med, lo, hi = [], [], []
        for gp in gaps:
            v = g[g.gap == gp].kept
            med.append(v.median())
            a, b = boot_median_ci(v, rng)
            lo.append(a)
            hi.append(b)
        assert np.allclose(med, imp.loc[gaps, f"{cond}_kept_median"], atol=1e-12)
        x = np.arange(4) + (k - 1.5) * 0.07
        med, lo, hi = map(np.array, (med, lo, hi))
        ax.errorbar(x, med, yerr=[med - lo, hi - med], color=col, ls=ls, lw=lw, marker=mk,
                    ms=3.5, mfc=col if ls == "-" else "white", mec=col, capsize=0, label=lab)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axhline(1, color=INK2, lw=0.6, ls="--")
    ax.set_xticks(range(4), ["≤ 7", "8–30", "31–180", "> 180"])
    ax.set_ylim(-0.15, 1.05)
    ax.set_xlabel("days apart")
    ax.set_ylabel("share of own R² kept (median, 95% CI)")
    ax.legend(frameon=False, loc="lower left")
    style(ax, "Matching variants by days apart")
    for x, y, L in ((0.0, H - 0.02, "A"), (0.0, 3.4, "B"), (3.5, 3.4, "C")):
        letter(fig, x, y, L)
    save(fig, "figS1_transfer")


# ------------------------------------------------------------------------------ Figure S2
def fig_embeddings():
    from src.evaluate import DATA_ROOT, SubsetPoyoMPDataset

    H = 5.15
    fig = figure(H)
    ax = axes(fig, 0.5, 3.05, 2.6, 1.75)
    up = read("unit_properties.csv")
    row = (up[up.well_tuned.astype(bool) & (up.task == "CO")]
           .sort_values("tuning_r2_100").iloc[-1])
    data = SubsetPoyoMPDataset(str(DATA_ROOT), [row.session]).get_recording(row.session)
    tuning_curve_panel(ax, data, row, "", data_color=BLUE, model_color=GREY)
    for ln in ax.get_lines():
        ln.set_linewidth(0.7 if ln.get_linestyle() == ':' else 1.2)
        ln.set_markersize(3)
    for c in ax.collections:
        c.set_linewidth(0.8)
    ax.tick_params(labelsize=6.5, colors=INK2)
    ax.set_ylabel("firing rate (Hz)")
    ax.set_xlabel("movement direction (deg)")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    ax.legend(frameon=False, loc="upper left")
    ax.text(np.degrees(row.pref_dir) - 4, 5.5, "preferred\ndirection", color=INK2,
            fontsize=FS_SMALL, va="center", ha="right")
    style(ax, "What the probes predict: one unit's tuning")

    ax = axes(fig, 4.25, 3.05, 2.35, 1.75)
    cv = read("emb_probe_scores.csv")
    cv = cv[cv.subset == "all"].set_index("target")
    names = [("log10_rate", "firing rate"), ("tuning_r2_100", "tuning strength")]
    y = np.arange(len(names))[::-1]
    sc = cv.loc[[n for n, _ in names]]
    ax.barh(y, sc.score, height=0.5, color=BLUE, label="held-out units (95% CI)")
    ax.errorbar(sc.score, y, xerr=[sc.score - sc.ci_lo, sc.ci_hi - sc.score], fmt="none",
                ecolor=INK2, elinewidth=0.8, capsize=0)
    ax.scatter(sc.session_mean_baseline, y, marker="|", s=120, color=INK, linewidths=1.3,
               zorder=4, label="session-mean oracle")
    ax.scatter(sc.chance95, y, marker="|", s=120, color=LIGHT, linewidths=1.3, zorder=4,
               label="shuffled targets (95th pct)")
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(y, [lab for _, lab in names])
    ax.set_xlim(-0.05, 0.7)
    ax.set_ylim(-0.8, 1.6)
    ax.set_xlabel("probe R² (held-out electrodes)")
    ax.legend(frameon=False, loc="lower right")
    style(ax, "Unit properties read from the embeddings", grid="x")

    panel_pref_dir(axes(fig, 0.6, 0.42, 1.95, 1.95), "Preferred direction, {n:,} held-out units")

    ax = axes(fig, 3.98, 0.42, 2.62, 1.95)
    pairs = read("emb_same_electrode_pairs.csv")
    edges = [0, 7, 30, 90, 180, 365, 730, 1200]
    for mk, ls, mfc in (("C", "-", BLUE), ("M", "--", "white")):
        g = pairs[(pairs.monkey == mk) & (pairs.kind == "observed")]
        m = g.groupby(pd.cut(g.gap_days, edges), observed=True)["diff"].agg(["mean", "sem"])
        xc = [iv.mid for iv in m.index]
        ax.errorbar(xc, m["mean"], yerr=m["sem"], color=BLUE, ls=ls, lw=1.3, marker="o", ms=3.5,
                    mfc=mfc, mec=BLUE, capsize=0)
        ax.text(xc[-1] * 1.15, m["mean"].iloc[-1], f"monkey {mk}", color=INK2,
                fontsize=FS_SMALL, va="center")
    g = pairs[pairs.kind == "shuffled"]
    m = g.groupby(pd.cut(g.gap_days, edges), observed=True)["diff"].mean()
    ax.plot([iv.mid for iv in m.index], m.values, color=LIGHT, lw=1.3, ls=":")
    ax.text(3.5, 0.015, "channel labels shuffled", color=INK2, fontsize=FS_SMALL, va="bottom")
    ax.axhline(0, color=INK2, lw=0.6)
    log_ticks(ax, [3, 10, 30, 100, 300, 1000])
    ax.set_xlim(2, 3500)
    ax.set_xlabel("days between the two sessions")
    ax.set_ylabel("cosine similarity,\nsame − different electrode")
    style(ax, "Same electrode, more similar embeddings")
    for x, y, L in ((0.0, 5.12, "A"), (3.55, 5.12, "B"), (0.0, 2.66, "C"), (3.5, 2.66, "D")):
        letter(fig, x, y, L)
    save(fig, "figS2_unit_embeddings")


def tuning_curve_panel(ax, data, row, title, data_color=BLUE, model_color=ORANGE):
    rate, vel, edges = up.bin_recording(data)
    # main scope, built as in scripts/embeddings/unit_properties.py: train time during movement
    mov, _ = up.moving_intervals(edges, vel, up.used_intervals(data, "train", "scored"))
    iv = up.used_intervals(data, "train") & mov
    j = up.scored_bins(edges, iv, vel)
    L = int(round(row.best_lag_ms / 20))
    v = vel[j + L]
    y = rate[j, int(row.local_index)]
    th = np.arctan2(v[:, 1], v[:, 0])
    pred = row.b0 + row.bx * v[:, 0] + row.by * v[:, 1]
    bins = np.linspace(-np.pi, np.pi, 17)
    k = np.clip(np.digitize(th, bins) - 1, 0, 15)
    c = np.degrees((bins[:-1] + bins[1:]) / 2)
    mu = np.array([y[k == i].mean() for i in range(16)])
    se = np.array([y[k == i].std(ddof=1) / np.sqrt((k == i).sum()) for i in range(16)])
    pm = np.array([pred[k == i].mean() for i in range(16)])
    ax.errorbar(c, mu, yerr=se, color=data_color, lw=2, marker="o", ms=5, mec="white", capsize=0,
                label="data (mean ± SEM)")
    ax.plot(c, pm, color=model_color, lw=2, ls="--", label="linear velocity model")
    ax.axvline(np.degrees(row.pref_dir), color=INK2, lw=1, ls=":")
    ax.set_title(title, color=INK, fontsize=9, loc="left")
    ax.set_xlabel("movement direction at best lag (deg)", color=INK2)
    ax.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)


# ------------------------------------------------------------------------------ Figure S4
def fig_lesions():
    lv = read("lesion_vs_ridge.csv")
    lv = lv[lv.variant == "grid"].set_index("condition").loc[[c for c, _ in CONDS]]
    H = 4.15
    fig = figure(H)
    md.lesion_panel(canvas(fig, 0.15, 2.5, 6.6, 1.45), 6.6, 1.45)
    ax = axes(fig, 0.62, 0.38, 2.6, 1.78)
    panel_remove_silence(ax, "Absent beats silent")
    ax2 = axes(fig, 3.72, 0.38, 2.88, 1.78, sharey=ax)
    h, lab = dumbbell(ax2, COND_Y, 100 * lv.poyo_remove_rel_loss, 100 * lv.ridge_refit_rel_loss,
                      BLUE, GREY, "POYO, units removed, no retraining",
                      "ridge refitted on kept units", mb="s")
    ax2.set_xlim(0, 45)
    ax2.tick_params(labelleft=False)
    ax2.set_xlabel("loss (% of the decoder's own R²)")
    ax2.legend(h, lab, frameon=False, loc="upper right")
    style(ax2, "POYO vs a refitted decoder", grid="x")
    for x, y, L in ((0.0, H - 0.05, "A"), (0.0, 2.48, "B"), (3.55, 2.48, "C")):
        letter(fig, x, y, L)
    save(fig, "figS4_missing_neurons")


# ------------------------------------------------------------------------------ Figure S5
def fig_timing():
    sw = read("summary_sweep.csv")
    rel = read("binned_ridge_summary.csv")
    H = 4.95
    fig = figure(H)
    md.timing_panel(canvas(fig, 0.15, 2.95, 3.2, 1.8), 3.2, 1.8)

    ax = axes(fig, 4.12, 3.05, 2.5, 1.62)
    small = [1, 2, 5, 10, 20, 50]
    for (exp, col), (task, ls, mfc), off in zip(
            [(e, c) for e, c in (("bin_shuffle", BLUE), ("snap", ORANGE)) for _ in (0, 1)],
            [t for _ in (0, 1) for t in (("CO", "-", None), ("RT", "--", "white"))],
            (0.94, 0.98, 1.02, 1.06)):
        g = sw[(sw.experiment == exp) & (sw.task == task) & sw.bin_ms.isin(small)]
        g = g.sort_values("bin_ms")
        ax.errorbar(g.bin_ms * off, -1000 * g.dr2_mean, yerr=1000 * g.dr2_sem, color=col,
                    ls=ls, lw=1.2, marker="o", ms=3, mfc=mfc or col, mec=col, capsize=0,
                    label=f"{exp.replace('_', ' ')}, {task}")
    ax.axhline(1, color=INK2, lw=0.6, ls=":")
    ax.text(1.0, 1.12, "0.001 R²", color=INK2, fontsize=FS_SMALL)
    ax.axhline(0, color=INK2, lw=0.6)
    log_ticks(ax, small)
    ax.set_xlabel("bin size (ms)")
    ax.set_ylabel("R² loss (× 10⁻³)")
    ax.legend(frameon=False, loc="upper left")
    style(ax, "Fine timing: tiny losses")

    for k, (task, n) in enumerate((("CO", 78), ("RT", 21))):
        ax = axes(fig, 0.55 + k * 3.43, 0.4, 2.62, 1.85)
        for dec, col, ls, mk, lab in (
                ("POYO bin shuffle", BLUE, "-", "o", "POYO, bin shuffle"),
                ("POYO snap", ORANGE, "-", "o", "POYO, snap"),
                ("ridge (grid)", GREY, "--", "s", "ridge, trained per bin size")):
            g = rel[(rel.decoder == dec) & (rel.task == task)].sort_values("bin_ms")
            ax.errorbar(g.bin_ms, 100 * g.frac_loss, yerr=100 * g.frac_loss_sem, color=col,
                        ls=ls, lw=1.3, marker=mk, ms=3.5, mec="white", mew=0.5, capsize=0,
                        label=lab)
        latent_line(ax, 52)
        ax.axhline(0, color=INK2, lw=0.6)
        log_ticks(ax, [10, 20, 50, 100, 200, 500])
        ax.set_ylim(-5, 85)
        ax.set_xlabel("bin size (ms)")
        ax.set_ylabel("loss (% of the decoder's own R²)")
        if k == 0:
            ax.legend(frameon=False, loc="upper left")
        style(ax, f"Coarse bins: {task} ({n} sessions)")
    for x, y, L in ((0.0, 4.93, "A"), (3.55, 4.93, "B"), (0.0, 2.55, "C"), (3.43, 2.55, "D")):
        letter(fig, x, y, L)
    save(fig, "figS5_spike_timing")


# ------------------------------------------------------------------------------ Figure S3
LETTER_FAMILIES = [  # (label, conditions in bit order, line style, marker)
    ("one alphabet", [f"kmeans_K{k}" for k in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)], "-", "o"),
    ("4 parallel letters", ["product_M4_K4", "product_M4_K8", "product_M4_K16", "product_M4_K64"],
     "--", "s"),
    ("coarse-to-fine, 16 letters per stage", [f"residual_16x{n}" for n in range(1, 7)], ":", "^"),
]
SCHEME_COLOR = {"sessions": BLUE, "monkeys": ORANGE}
SCHEME_LABEL = {"sessions": "codebook from other sessions", "monkeys": "codebook from other monkeys"}


def letters_panel(ax, summ, scheme, title):
    col = SCHEME_COLOR[scheme]
    for k, (lab, conds, ls, mk) in enumerate(LETTER_FAMILIES):
        g = summ.loc[[(scheme, c) for c in conds]]
        x = g.bits.to_numpy() + (k - 1) * 0.12
        med = g.kept_median.to_numpy()
        ax.errorbar(x, med, yerr=[med - g.kept_ci_lo, g.kept_ci_hi - med], color=col, ls=ls,
                    lw=1.1, marker=mk, ms=3.2, mfc=col if k == 0 else "white", mec=col,
                    capsize=0, label=lab)
    nulls = summ.loc[[(scheme, c) for c in ("area", "random_K16", "random_K256")]]
    ax.scatter(nulls.bits, nulls.kept_median, marker="x", s=14, color=LIGHT, lw=1.0,
               label="area / random letters", zorder=3)
    pc = summ.loc[(scheme, "pca_d16"), "kept_median"]
    ax.axhline(pc, color=INK2, lw=0.7, ls=(0, (1, 1.5)))
    ax.text(19.5, pc - 0.015, "16 continuous dims", fontsize=FS_SMALL, color=INK2, ha="right",
            va="top")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axhline(1, color=INK2, lw=0.6, ls="--")
    ax.set_xlim(-0.5, 25)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_ylim(-0.08, 1.05)
    ax.set_xlabel("code length (bits)")
    ax.set_ylabel("share of own R² kept (median, 95% CI)")
    style(ax, title)


def fig_letters():
    summ = read("codebook_summary.csv")
    summ = summ[summ.monkey == "all"].set_index(["scheme", "condition"])
    H = 5.55
    fig = figure(H)
    ax = axes(fig, 0.55, 3.3, 2.75, 2.0)
    letters_panel(ax, summ, "sessions", "Codebook from other sessions")
    ax.legend(frameon=False, loc="lower right", fontsize=FS_SMALL)
    letters_panel(axes(fig, 3.85, 3.3, 2.75, 2.0), summ, "monkeys",
                  "Codebook from other monkeys")

    ax = axes(fig, 0.55, 0.77, 2.75, 1.95)
    dims = [1, 2, 4, 8, 16, 32, 64]
    for sc in ("sessions", "monkeys"):
        g = summ.loc[[(sc, f"pca_d{d}") for d in dims]]
        med = g.kept_median.to_numpy()
        ax.errorbar(dims, med, yerr=[med - g.kept_ci_lo, g.kept_ci_hi - med],
                    color=SCHEME_COLOR[sc], marker="o", ms=3.2, mec="white", mew=0.5, capsize=0,
                    label=f"axes from other {sc}")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axhline(1, color=INK2, lw=0.6, ls="--")
    log_ticks(ax, dims)
    ax.set_ylim(-0.08, 1.05)
    ax.set_xlabel("principal components kept")
    ax.set_ylabel("share of own R² kept (median, 95% CI)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    style(ax, "Continuous control: top principal components")

    ax = axes(fig, 4.75, 0.77, 1.85, 1.95)
    rows = [("kmeans_K16", "16 letters"), ("kmeans_K256", "256 letters"),
            ("kmeans_K1024", "1024 letters"), ("product_M4_K64", "4 parallel × 64"),
            ("residual_16x6", "6 stages × 16"), ("residual_256x3", "3 stages × 256")]
    have = ("sessions", "dir_kmeans_K256") in summ.index
    y = np.arange(len(rows))[::-1].astype(float)
    for k, sc in enumerate(("sessions", "monkeys")):
        col, yy = SCHEME_COLOR[sc], y + (0.17 if k == 0 else -0.17)
        whole = [summ.loc[(sc, c), "kept_median"] for c, _ in rows]
        for i, (c, _) in enumerate(rows):
            if have:
                ax.plot([whole[i], summ.loc[(sc, f"dir_{c}"), "kept_median"]], [yy[i]] * 2,
                        color=LIGHT, lw=1.4, zorder=1)
        ax.scatter(whole, yy, s=14, facecolors="white", edgecolors=col, lw=0.9, zorder=3,
                   label=f"whole vector ({sc})")
        if have:
            ax.scatter([summ.loc[(sc, f"dir_{c}"), "kept_median"] for c, _ in rows], yy, s=14,
                       color=col, edgecolors="white", lw=0.5, zorder=4,
                       label=f"direction only ({sc})")
    ax.axvline(0, color=INK2, lw=0.6)
    ax.axvline(1, color=INK2, lw=0.6, ls="--")
    ax.set_yticks(y, [lab for _, lab in rows])
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("share of own R² kept (median)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.3, -0.25), ncol=2,
              fontsize=FS_SMALL - 0.5, handletextpad=0.2, columnspacing=0.6)
    style(ax, "Letters on direction, own length", grid="x")
    for x, yv, L in ((0.0, H - 0.02, "A"), (3.35, H - 0.02, "B"), (0.0, 2.97, "C"),
                     (3.55, 2.97, "D")):
        letter(fig, x, yv, L)
    save(fig, "figS3_neuron_letters")


if __name__ == "__main__":
    fig_results()
    fig_transfer()
    fig_embeddings()
    fig_lesions()
    fig_timing()
    fig_letters()
