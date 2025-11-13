from typing import List, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from trident.Converter import OPENSLIDE_EXTENSIONS, PIL_EXTENSIONS

from nova.utils.deterministic import _set_deterministic

# Constants

VALID_SLIDE_EXTS = OPENSLIDE_EXTENSIONS.union(PIL_EXTENSIONS)

# Set deterministic behavior
_set_deterministic()


__all__ = [
    "plot_cv_indices",
    "plot_manual_cv_assignments",
]


def plot_manual_cv_assignments(
    assignments: List[np.ndarray], y: np.ndarray, line_width: int = 10
) -> Tuple[Figure, Axes]:
    """
    Visualize manual cross-validation split assignments and sample classes.

    Args:
        assignments: List of arrays indicating train (0), test (1), or np.nan for each sample.
        y: True class labels for each sample.
        line_width: Line width for plot markers.

    Returns:
        Tuple of matplotlib Figure and Axes with CV assignments visualized.
    """

    N = len(assignments[0])
    n_splits = len(assignments)
    fig, ax = plt.subplots()

    for ii, indices in enumerate(assignments):
        ax.scatter(
            range(1, N + 1),
            [ii + 0.5] * N,
            c=indices,
            marker="_",
            lw=line_width,
            cmap="coolwarm",
            vmin=-0.2,
            vmax=1.2,
        )
    # Plot classes row
    ax.scatter(range(1, N + 1), [n_splits + 0.5] * N, c=y, marker="_", lw=line_width, cmap="Paired")
    yticklabels = list(range(n_splits)) + ["class"]
    ax.set(
        yticks=np.arange(n_splits + 1) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 0.8, -0.2],
        xlim=[1, N],
    )
    step = max(1, N // 10)
    xticks = list(range(1, N + 1, step))
    if N not in xticks:
        xticks.append(N)
    ax.set_xticks(xticks)

    train_proxy = mlines.Line2D([], [], color="blue", marker="_", linestyle='None', markersize=15, label="Train")
    test_proxy = mlines.Line2D([], [], color="red", marker="_", linestyle='None', markersize=15, label="Test")
    ax.legend(handles=[train_proxy, test_proxy], loc="upper right", title="Split")
    return fig, ax


def plot_cv_indices(cv, X: np.ndarray, y: np.ndarray, ax: Axes, n_splits: int, lw: int = 10) -> Axes:
    """
    Visualize cross-validation splits and class labels.

    Args:
        cv: Cross-validation splitter supporting .split(X, y).
        X: Feature data for splitting.
        y: Class labels.
        ax: Matplotlib Axes to plot on.
        n_splits: Number of cross-validation splits.
        lw: Line width for markers.

    Returns:
        The input Axes with split assignments plotted.
    """

    # Generate the training/testing visualizations for each CV split
    ii = 0
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(1, len(indices) + 1),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap="coolwarm",
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes
    ax.scatter(range(1, len(X) + 1), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap="Paired")

    # Formatting
    N = len(X)
    yticklabels = list(range(n_splits)) + ["class"]
    ax.set(
        yticks=np.arange(n_splits + 1) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 0.8, -0.2],
        xlim=[1, N],  # Starts at 1, ends at N
    )

    # Set x-ticks at regular intervals, including first and last sample
    step = max(1, N // 10)
    xticks = list(range(1, N + 1, step))
    if N not in xticks:
        xticks.append(N)
    ax.set_xticks(xticks)

    train_proxy = mlines.Line2D([], [], color="blue", marker="_", linestyle='None', markersize=15, label="Train")
    test_proxy = mlines.Line2D([], [], color="red", marker="_", linestyle='None', markersize=15, label="Test")
    ax.legend(handles=[train_proxy, test_proxy], loc="upper right", title="Split")

    return ax
