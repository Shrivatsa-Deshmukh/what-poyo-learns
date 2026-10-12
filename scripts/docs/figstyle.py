"""Shared style for every report figure.

Figures are laid out in inches at their printed size (full text width 6.75 in), so a 7 pt label
prints at 7 pt. Each figure is saved as a vector PDF and a 300 dpi PNG.

Colours have one meaning in every figure:
  BLUE    POYO, main condition (matched electrodes, remove, bin shuffle)
  ORANGE  POYO, second condition (silence, snap)
  GREY    comparison decoder without POYO (ridge), dashed
  LIGHT   shuffles and nulls
  UNIT    one colour per unit / electrode in the schematics
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "figures"
TEXT_W = 6.75  # in

BLUE, ORANGE, GREY, LIGHT = "#2a78d6", "#eb6834", "#8a8984", "#c4c3be"
BLUE_LIGHT = "#8db8ea"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BOX_FILL, BOX_EDGE = "#f3f3f1", "#b5b4ae"
UNIT = ["#7a5cc4", "#2aa198", "#a0522d", "#d1478a"]

FS_TICK, FS_LABEL, FS_TITLE, FS_LETTER, FS_SMALL = 6.5, 7, 7.5, 9.5, 6

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": FS_LABEL, "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_TITLE, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_SMALL + 0.5, "legend.handlelength": 1.8, "legend.borderaxespad": 0.3,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "axes.labelpad": 2.5, "lines.linewidth": 1.3, "lines.markersize": 3.5,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


# --------------------------------------------------------------------------- layout in inches
def figure(h, w=TEXT_W):
    fig = plt.figure(figsize=(w, h), facecolor="white")
    fig._inch = (w, h)
    return fig


def axes(fig, x, y, w, h, **kw):
    """Axes at (x, y) with size (w, h), all in inches from the figure's bottom-left corner."""
    W, H = fig._inch
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def canvas(fig, x, y, w, h):
    """Axes for drawing schematics: data coordinates are inches inside the axes, no frame."""
    ax = axes(fig, x, y, w, h)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return ax


def letter(fig, x, y, s):
    """Panel letter with its top-left corner at (x, y) inches."""
    W, H = fig._inch
    fig.text(x / W, y / H, s, fontsize=FS_LETTER, weight="bold", color=INK, va="top", ha="left")


def style(ax, title=None, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(True, axis=grid, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=FS_TITLE, loc="left", pad=4)


def log_ticks(ax, ticks, axis="x"):
    ax.set_xscale("log") if axis == "x" else ax.set_yscale("log")
    getattr(ax, f"set_{axis}ticks")(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()


def save(fig, name):
    """Save figures/<name>.pdf (vector) and <name>.png (300 dpi) at the exact layout size."""
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white", **kw)
    plt.close(fig)
    print("saved", OUT / f"{name}.pdf", "and .png")


# ------------------------------------------------------------------------ schematic primitives
def box(ax, x, y, w, h, fc=BOX_FILL, ec=BOX_EDGE, lw=0.8, r=0.04, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def text(ax, x, y, s, size=FS_SMALL, color=INK2, weight="normal", ha="left", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            linespacing=1.25, zorder=6, **kw)


def arrow(ax, p, q, color=INK2, lw=0.8, head=6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=4))


def token(ax, x, y, color, kind="spike", s=0.1, lost=False, ec=None, lw=0.7):
    """Square token centred at (x, y): 'spike' filled in the unit's colour, 'delim' ([START] /
    [END]) open with a unit-colour edge; lost = dashed empty slot (token deleted)."""
    if lost:
        fc, edge, ls = "white", LIGHT, (0, (1.5, 1.2))
    elif kind == "spike":
        fc, edge, ls = color, color, "-"
    else:
        fc, edge, ls = "white", color, "-"
    ax.add_patch(FancyBboxPatch((x - s / 2, y - s / 2), s, s,
                                boxstyle=f"round,pad=0,rounding_size={s * 0.18}", fc=fc,
                                ec=ec or edge, lw=lw, ls=ls, zorder=5))


ALPHA = (1, .55, .8, .35, .9, .6)


def vector(ax, x, y, color, n=6, w=0.09, h=0.15, alpha=ALPHA):
    """A learned embedding drawn as n cells of one colour with varying shade."""
    for i in range(n):
        ax.add_patch(plt.Rectangle((x + i * w, y - h / 2), w, h, fc=color,
                                   alpha=alpha[i % len(alpha)], ec="white", lw=0.4, zorder=5))


def ticks(ax, x0, x1, y, times, color, h=0.08, lw=1.0):
    for t in times:
        xx = x0 + t * (x1 - x0)
        ax.plot([xx, xx], [y - h, y + h], color=color, lw=lw, solid_capstyle="butt", zorder=5)


def marker(ax, x, y, n, r=0.075):
    """Numbered marker: where analysis n (Figure 2 panel n) acts on the model."""
    ax.add_patch(plt.Circle((x, y), r, fc=INK, ec="white", lw=0.8, zorder=8))
    ax.text(x, y - 0.004, str(n), fontsize=6.5, color="white", fontweight="bold", ha="center",
            va="center", zorder=9)
