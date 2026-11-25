"""One look for every static figure in reports/figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402

# Series colours follow the role of the series, the same in every figure.
NAIVE = "#eb6834"  # the intuitive method (resolved tickets only, or a fixed average)
SURVIVAL = "#2a78d6"  # the survival-analysis estimate
OTHER = "#6f6e69"  # a method that is neither (the ordinary classifier)
TRUTH = "#0b0b0b"  # what actually happened

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

LANGS = ("tr", "en")


def apply() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 1,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "ytick.left": False,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
        }
    )


def line(ax, x, y, color, label, markers: bool = True, **kwargs):
    """A 2px line with ringed markers."""
    style = dict(color=color, label=label, linewidth=2)
    if markers:
        style.update(marker="o", markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1)
    style.update(kwargs)
    return ax.plot(x, y, **style)


def end_label(ax, x, y, text, dy: float = 0.0, dx: float = 6):
    """Value label at the end of a line, in text ink (never the series colour)."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=9,
        color=INK,
        annotation_clip=False,
    )


def save(fig, name: str, lang: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}_{lang}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(config.ROOT)}")
