"""Plot style and helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # save plots to files without opening a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Line colors
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

# one consistent look for every figure
plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "legend.labelcolor": INK2,
    "lines.linewidth": 2,
})


def moving_average(x: np.ndarray, w: int = 100) -> np.ndarray:
    """Moving average over the last w points."""
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))  # running totals
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i - w + 1)  # early points average whatever exists so far
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def band(ax, x, runs, color, label, smooth=100):
    """Plot the mean across seeds with a std band."""
    curves = np.array([moving_average(r, smooth) if smooth else r for r in runs])
    mean, std = curves.mean(0), curves.std(0)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)  # shaded spread
    ax.plot(x, mean, color=color, label=label)                                       # average line
    return mean


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)  # free memory between plots
