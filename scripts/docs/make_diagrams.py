"""Report Figure 1 and the schematic panels of Figures S1, S4 and S5 (no data needed).

figure1()            figures/fig1_overview.{pdf,png}: POYO-MP and the four analyses
transfer_panel(ax)   Figure S1A: another day's same-electrode embeddings (matched / shuffled)
lesion_panel(ax)     Figure S4A: remove vs silence, as tokens
timing_panel(ax)     Figure S5A: snap / bin shuffle / window shuffle

The panel functions draw into a canvas axes (data coordinates in inches, scripts/docs/figstyle.py)
of the figure that uses them (scripts/docs/paper_figures.py). Style and colours: figstyle.py.
Usage: python scripts/docs/make_diagrams.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (  # noqa: E402
    BLUE, BOX_EDGE, FS_LETTER, FS_SMALL, FS_TITLE, GREY, INK, INK2, LIGHT, ORANGE, UNIT, arrow,
    box, canvas, figure, marker, save, text, ticks, token, vector,
)

SPIKES = [[0.08, 0.3, 0.45, 0.8], [0.2, 0.62], [0.05, 0.27, 0.5, 0.66, 0.9]]


def heading(ax, x, y, letter, title):
    ax.text(x, y, letter, fontsize=FS_LETTER, weight="bold", color=INK, va="top", ha="left")
    ax.text(x + 0.19, y - 0.015, title, fontsize=FS_TITLE, color=INK, va="top", ha="left")


# ------------------------------------------------------------------------------- Figure 1
def figure1():
    """(A) POYO-MP as used here, numbered markers where each analysis acts; (B-E) what each
    analysis changes, numbered and coloured as Figure 2."""
    H = 4.25
    fig = figure(H)
    ax = canvas(fig, 0.05, 0, 6.68, H)
    fs = FS_SMALL

    # ------------------------------------------------------------ A: the model
    heading(ax, 0.0, H - 0.02, "A", "POYO-MP: every spike becomes a token of its neuron")
    ys = [3.55, 3.27, 2.99]
    text(ax, 0.3, 3.82, "spikes, 1 s", size=fs)
    for y, u, c, sp in zip(ys, ("u1", "u2", "u3"), UNIT, SPIKES):
        text(ax, 0.02, y, u, size=fs, color=c, weight="bold")
        ticks(ax, 0.3, 1.1, y, sp, c)
    ax.plot([0.3, 1.1], [2.8, 2.8], color=INK2, lw=0.6)
    text(ax, 0.3, 2.7, "0", size=fs, ha="center")
    text(ax, 1.1, 2.7, "1 s", size=fs, ha="center")
    arrow(ax, (1.17, 3.27), (1.35, 3.27))
    text(ax, 1.42, 3.82, "unit embeddings", size=fs)
    for y, c in zip(ys, UNIT):
        vector(ax, 1.42, y, c)
    text(ax, 1.42, 2.72, "one vector\nper neuron", size=fs)
    arrow(ax, (2.0, 3.27), (2.17, 3.27))
    text(ax, 2.26, 3.82, "tokens", size=fs)
    x0, x1 = 2.43, 3.33
    for y, c, sp in zip(ys, UNIT, SPIKES):
        token(ax, 2.28, y, c, "delim", s=0.11)
        token(ax, 3.5, y, c, "delim", s=0.11)
        for t in sp:
            token(ax, x0 + t * (x1 - x0), y, c, s=0.09)
    text(ax, 2.26, 2.72, "time = spike time;\nopen = [START] / [END]", size=fs)
    arrow(ax, (3.62, 3.27), (3.8, 3.27))
    yb, hb = 2.88, 0.95
    for x, t, b in ((3.8, "Encoder", "latents read\nall tokens"),
                    (4.62, "Processor", "24 self-\nattention\nlayers"),
                    (5.44, "Decoder", "queries read\nthe latents")):
        if t == "Processor":
            box(ax, x + 0.05, yb + 0.05, 0.68, hb)
            box(ax, x + 0.025, yb + 0.025, 0.68, hb)
        box(ax, x, yb, 0.68, hb)
        text(ax, x + 0.06, yb + hb - 0.1, t, size=6.5, color=INK, weight="bold")
        text(ax, x + 0.06, yb + hb - 0.33, b, size=fs)
    for i in range(8):
        ax.add_patch(plt_circle((3.87 + i * 0.075, 2.98), 0.022))
    text(ax, 3.86, 3.1, "8 time points", size=5.5)
    arrow(ax, (4.48, 3.27), (4.62, 3.27))
    arrow(ax, (5.35, 3.27), (5.44, 3.27))
    ax.plot([3.8, 3.8, 6.12, 6.12], [3.9, 3.94, 3.94, 3.9], color=INK2, lw=0.6)
    text(ax, 4.96, 4.02, "frozen; time enters only inside attention", size=fs, ha="center",
         va="bottom")
    box(ax, 5.44, 2.42, 0.68, 0.3)
    text(ax, 5.5, 2.63, "session emb.", size=fs, color=INK)
    text(ax, 5.5, 2.5, "+ output time", size=5.5)
    arrow(ax, (5.78, 2.72), (5.78, 2.88))
    arrow(ax, (6.12, 3.27), (6.19, 3.27))
    tt = np.linspace(0, 1, 50)
    for k, col in enumerate((INK, INK2)):
        ax.plot(6.2 + 0.34 * tt, 3.38 - 0.2 * k + 0.06 * np.sin(2 * np.pi * (1.2 * tt + k / 5)),
                color=col, lw=0.9, zorder=5)
    text(ax, 6.56, 3.38, "vx", size=5.5, color=INK)
    text(ax, 6.56, 3.18, "vy", size=5.5)
    text(ax, 6.2, 3.62, "velocity", size=fs)
    marker(ax, 1.42, 3.72, 1)
    marker(ax, 1.96, 3.72, 2)
    marker(ax, 2.1, 2.99, 3)
    marker(ax, 0.7, 2.8, 4)
    ax.plot([0, 6.75], [2.3, 2.3], color="#e4e3df", lw=0.6)

    # ------------------------------------------------------------ B-E: the analyses
    top = 2.18
    xs = [0.0, 1.71, 3.42, 5.1]
    for x, L, t in zip(xs, "BCDE", ("1 · Another day", "2 · Read the embeddings",
                                    "3 · Missing neurons", "4 · Moved spike times")):
        heading(ax, x, top, L, t)
    rows = [1.6, 1.2, 0.8]

    # B: same-electrode swap
    x = xs[0]
    text(ax, x + 0.2, 1.88, "other day", size=fs)
    text(ax, x + 1.02, 1.88, "this day", size=fs)
    for i, (y, e) in enumerate(zip(rows, ("e1", "e2", "e3"))):
        text(ax, x, y, e, size=fs)
        if i == 0:
            vector(ax, x + 0.2, y + 0.08, UNIT[0], n=4, h=0.12)
            vector(ax, x + 0.2, y - 0.08, UNIT[0], n=4, h=0.12, alpha=(.7, 1, .5, .8))
        elif i == 1:
            vector(ax, x + 0.2, y, UNIT[1], n=4, h=0.12)
        else:
            text(ax, x + 0.2, y, "none", size=fs)
        vector(ax, x + 1.02, y, BOX_EDGE if i == 2 else UNIT[i], n=4, h=0.12,
               alpha=(1, 1, 1, 1) if i == 2 else (.85, .78, .65, .8))
        arrow(ax, (x + 0.6, y), (x + 0.97, y), color=BLUE)
    text(ax, x, 0.42, "same-electrode average;\nempty: other day's average", size=fs)
    text(ax, x, 0.1, "control: random electrodes", size=fs, color=GREY)

    # C: linear probe
    x = xs[1]
    vector(ax, x, rows[0], UNIT[2])
    arrow(ax, (x + 0.57, rows[0]), (x + 0.72, rows[0]), color=BLUE)
    box(ax, x + 0.74, rows[0] - 0.15, 0.62, 0.3, fc="white")
    text(ax, x + 1.05, rows[0], "linear probe", size=fs, color=INK, ha="center")
    arrow(ax, (x + 0.9, rows[0] - 0.15), (x + 0.55, 1.12), color=BLUE)
    arrow(ax, (x + 1.2, rows[0] - 0.15), (x + 1.35, 1.12), color=BLUE)
    ax.add_patch(plt_circle((x + 0.45, 0.88), 0.19, fill=False))
    arrow(ax, (x + 0.45, 0.88), (x + 0.58, 1.01), color=BLUE, lw=1.0)
    for i, hgt in enumerate((0.08, 0.2, 0.13, 0.28)):
        ax.add_patch(plt_rect((x + 1.18 + i * 0.08, 0.72), 0.06, hgt))
    text(ax, x + 0.45, 0.6, "preferred\ndirection", size=fs, ha="center", va="top")
    text(ax, x + 1.32, 0.6, "firing\nrate", size=fs, ha="center", va="top")
    text(ax, x, 0.1, "tested on unseen electrodes", size=fs, color=GREY)

    # D: remove vs silence
    x = xs[2]
    seq = [("d", 0), ("d", 1), ("d", 2), ("s", 0), ("s", 2), ("s", 1), ("s", 2), ("d", 0),
           ("d", 1), ("d", 2)]
    pos = [x + 0.5 + k * 0.105 + (0.05 if k >= 3 else 0) + (0.05 if k >= 7 else 0)
           for k in range(len(seq))]
    for xc, lab in ((pos[1], "start"), ((pos[4] + pos[5]) / 2, "spikes"), (pos[8], "end")):
        text(ax, xc, 1.85, lab, size=fs, ha="center")
    for r, (y, (lab, col)) in enumerate(zip(rows, (("all", INK), ("remove", BLUE),
                                                    ("silence", ORANGE)))):
        text(ax, x, y, lab, size=6.5, color=col, weight="bold")
        for xx, (kind, u) in zip(pos, seq):
            lost = u == 2 and (r == 1 or (r == 2 and kind == "s"))
            token(ax, xx, y, UNIT[u], "spike" if kind == "s" else "delim", s=0.085, lost=lost)
    text(ax, x, 0.42, "u3 lost: remove (as in training)\nor silence (still listed)", size=fs)
    text(ax, x, 0.1, "reference: refitted ridge", size=fs, color=GREY)

    # E: snap / bin shuffle
    x = xs[3]
    g0, g1 = x + 0.62, x + 1.55
    for b in range(5):
        xx = g0 + b * (g1 - g0) / 4
        ax.plot([xx, xx], [0.66, 1.72], color="#e4e3df", lw=0.6, ls=(0, (2, 1.5)), zorder=1)
    orig = [0.04, 0.16, 0.2, 0.41, 0.57, 0.63, 0.92]
    shuf = [0.1, 0.03, 0.22, 0.33, 0.52, 0.71, 0.97]
    snap = [0.125, 0.125, 0.125, 0.375, 0.625, 0.625, 0.875]
    for y, lab, col, tt in ((rows[0], "original", INK, orig), (rows[1], "bin shuffle", BLUE, shuf),
                            (rows[2], "snap", ORANGE, snap)):
        text(ax, x, y, lab, size=6.5, color=col, weight="bold")
        ticks(ax, g0, g1, y, tt, col, h=0.09)
    text(ax, g0 + 0.125 * (g1 - g0), 0.97, "×3", size=5.5, color=ORANGE, ha="center")
    text(ax, g0 + 0.625 * (g1 - g0), 0.97, "×2", size=5.5, color=ORANGE, ha="center")
    text(ax, (g0 + g1) / 2, 1.85, "time bins", size=fs, ha="center")
    text(ax, x, 0.42, "spikes move within their bin;\ncounts never change", size=fs)
    text(ax, x, 0.1, "control: binned ridge", size=fs, color=GREY)
    save(fig, "fig1_overview")


def plt_circle(xy, r, fill=True):
    import matplotlib.pyplot as plt
    return plt.Circle(xy, r, fc=INK2 if fill else "none", ec=LIGHT if not fill else "none",
                      lw=0.7, zorder=5)


def plt_rect(xy, w, h):
    import matplotlib.pyplot as plt
    return plt.Rectangle(xy, w, h, fc=BLUE, alpha=0.85, ec="none", zorder=5)


# ------------------------------------------------------------------- supplementary schematics
def transfer_panel(ax, w, h):
    """Figure S1A in a canvas of w x h inches."""
    rows = [h - 0.52, h - 0.76, h - 1.0, h - 1.24]
    other = [2, 1, 3, 0]          # units per electrode on the other day
    target = [1, 2, 1, 1]         # units per electrode on this day
    shuffled = [2, 0, 1, None]    # electrodes re-paired at random (None: empty)

    def vecs(x, y, k, c):
        for i in range(k):
            vector(ax, x + i * 0.42, y, c, n=4, w=0.085, h=0.13)

    box(ax, 0.0, 0.05, 2.0, h - 0.1, fc="white", ec=BLUE, lw=0.7)
    text(ax, 0.08, h - 0.17, "other day", size=6.5, color=INK, weight="bold")
    for i, y in enumerate(rows):
        text(ax, 0.08, y, f"electrode {i + 1}", size=FS_SMALL)
        if other[i]:
            vecs(0.75, y, other[i], UNIT[i])
        else:
            text(ax, 0.75, y, "no unit recorded", size=FS_SMALL)
    box(ax, 2.55, 0.05, 2.55, h - 0.1, fc="white", ec=BLUE, lw=0.7)
    text(ax, 2.63, h - 0.17, "day being decoded", size=6.5, color=INK, weight="bold")
    text(ax, 3.62, h - 0.33, "matched", size=FS_SMALL, color=BLUE, weight="bold")
    text(ax, 4.37, h - 0.33, "shuffled", size=FS_SMALL, color=GREY, weight="bold")
    for i, y in enumerate(rows):
        text(ax, 2.63, y, f"electrode {i + 1}", size=FS_SMALL)
        vecs(3.62, y, 1, UNIT[i] if other[i] else BOX_EDGE)
        if target[i] > 1:
            vecs(3.62 + 0.0, y, 1, UNIT[i])
        s = shuffled[i]
        vecs(4.37, y, 1, UNIT[s] if s is not None else BOX_EDGE)
        if other[i]:
            arrow(ax, (0.75 + other[i] * 0.42 - 0.03, y), (2.53, y), color=BLUE, lw=0.7)
    text(ax, 2.27, h - 0.17, "average per\nelectrode", size=FS_SMALL, color=BLUE, ha="center",
         va="top")
    arrow(ax, (5.1, h / 2), (5.35, h / 2))
    box(ax, 5.35, 0.05, 1.38, h - 0.1)
    text(ax, 5.43, h - 0.17, "POYO-MP, frozen", size=6.5, color=INK, weight="bold")
    text(ax, 5.43, h / 2 - 0.05, "decodes this day's\ntest data with the\nswapped unit\nembeddings",
         size=FS_SMALL)


def lesion_panel(ax, w, h):
    """Figure S4A in a canvas of w x h inches."""
    seq = ([("s", u) for u in range(4)] + [("p", u) for u in (0, 2, 1, 2, 3, 0)]
           + [("e", u) for u in range(4)])
    s, step, gap, x0 = 0.15, 0.19, 0.12, 1.35
    pos, x = [], x0
    for i in range(len(seq)):
        pos.append(x)
        x += step + (gap if i in (3, 9) else 0)
    for xc, lab in ((np.mean(pos[0:4]), "[START]"), (np.mean(pos[4:10]), "spike tokens"),
                    (np.mean(pos[10:14]), "[END]")):
        text(ax, xc, h - 0.12, lab, size=FS_SMALL, ha="center")
    conds = [("all units", INK, set(), "14 tokens"),
             ("remove u3", BLUE, {"s", "p", "e"}, "10 tokens, as in training"),
             ("silence u3", ORANGE, {"p"}, "12 tokens, u3 still listed")]
    for r, (lab, col, gone, note) in enumerate(conds):
        y = h - 0.42 - r * 0.33
        text(ax, 0.05, y, lab, size=6.5, color=col, weight="bold")
        for xx, (kind, u) in zip(pos, seq):
            lost = u == 2 and kind in gone
            flag = u == 2 and kind in "se" and lab.startswith("silence")
            token(ax, xx, y, UNIT[u], "spike" if kind == "p" else "delim", s=s, lost=lost,
                  ec=ORANGE if flag else None, lw=1.3 if flag else 0.7)
        text(ax, pos[-1] + 0.25, y, note, size=FS_SMALL)
    for i, u in enumerate(("u1", "u2", "u3", "u4")):
        token(ax, 0.1 + i * 0.3, 0.12, UNIT[i], "spike", s=0.1)
        text(ax, 0.18 + i * 0.3, 0.12, u, size=FS_SMALL)
    text(ax, 1.35, 0.12, "open: [START] / [END], filled: spike token, dashed: deleted",
         size=FS_SMALL)


def timing_panel(ax, w, h):
    """Figure S5A in a canvas of w x h inches (20 ms bins over 100 ms)."""
    x0, x1 = 1.05, w - 0.12
    X = lambda ms: x0 + (x1 - x0) * ms / 100  # noqa: E731
    rows = [("original", INK, [3, 8, 17, 26, 44, 47, 71, 93]),
            ("snap", ORANGE, [10, 10, 10, 30, 50, 50, 70, 90]),
            ("bin shuffle", BLUE, [14, 2, 11, 33, 41, 58, 66, 97]),
            ("window shuffle", GREY, [12, 29, 35, 52, 63, 78, 81, 88])]
    ys = [h - 0.25 - k * (h - 0.6) / 3.4 for k in range(4)]
    for ms in range(0, 101, 20):
        ax.plot([X(ms), X(ms)], [0.3, h - 0.08], color="#e4e3df", lw=0.6, ls=(0, (2, 1.5)),
                zorder=1)
        text(ax, X(ms), 0.15, f"{ms}" + (" ms" if ms == 100 else ""), size=FS_SMALL,
             ha="center")
    ax.plot([x0, x1], [0.28, 0.28], color=INK2, lw=0.6)
    for y, (name, col, times) in zip(ys, rows):
        text(ax, 0.02, y, name, size=6.5, color=INK if col in (INK, GREY) else col,
             weight="bold")
        for t in times:
            ax.plot([X(t), X(t)], [y - 0.1, y + 0.1], color=col, lw=1.1, zorder=4)
    text(ax, X(10) + 0.04, ys[1] + 0.15, "×3", size=5.5, color=ORANGE)
    text(ax, X(50) + 0.04, ys[1] + 0.15, "×2", size=5.5, color=ORANGE)


if __name__ == "__main__":
    figure1()
