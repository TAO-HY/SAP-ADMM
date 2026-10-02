"""Read saved signal results and create SAP-ADMM figures."""

from __future__ import annotations

import argparse
from pathlib import Path

# Support running this file directly in Spyder or with python /path/to/file.py.
# Resolve imports from this checkout independently of the working directory.
if __package__ in (None, ""):
    import sys
    _project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_project_root))
    sys.path.insert(0, str(_project_root / "src"))
    __package__ = "experiments"

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

from sap_admm.utils import sample_std
from ._shared import REPOSITORY_ROOT, configure_plotting, save_figure


PAPER_LABELS = {
    "sap_admm": r"$\mathbf{SAP}$-$\mathbf{ADMM}$",
    "sap_admm_halpern": r"$\mathbf{SAP}$-$\mathbf{ADMM}^{\mathbf{H}}$",
}
STYLES = {
    "sap_admm": ("#0072B2", "-", "o"),
    "sap_admm_halpern": ("#4D4D4D", "--", "X"),
}
FILL_COLORS = {"sap_admm": "#B3D9FF", "sap_admm_halpern": "#D9D9D9"}


def _probability_labels(probabilities) -> list[str]:
    return ["0" if value == 0 else f"{value:.2f}" for value in probabilities]


def _panel_label(ax, label: str, *, fontsize: float = 26, y: float = -0.25) -> None:
    ax.text(0.5, y, label, transform=ax.transAxes, ha="center", va="top",
            fontsize=fontsize, fontweight="bold")


def _comparison(data, figure_dir: Path) -> None:
    probabilities, methods = data["probabilities"], list(data["methods"])
    mean_f1 = data["f1"].mean(0)
    std_f1 = sample_std(data["f1"])
    mean_time = data["seconds"].mean(0)
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8))
    for k, method in enumerate(methods):
        color, linestyle, marker = STYLES[str(method)]
        options = dict(color=color, linestyle=linestyle, marker=marker,
                       linewidth=2.2, markersize=7, markerfacecolor="white",
                       markeredgewidth=1.3, label=PAPER_LABELS[str(method)])
        axes[0].fill_between(
            probabilities, mean_f1[:, k] - std_f1[:, k], mean_f1[:, k] + std_f1[:, k],
            color=FILL_COLORS[str(method)], alpha=0.22 if k == 0 else 0.18,
            linewidth=0,
        )
        axes[0].plot(probabilities, mean_f1[:, k], **options)
        axes[1].plot(probabilities, mean_time[:, k], **options)
        axes[2].plot(probabilities, mean_f1[:, k] / mean_time[:, k], **options)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_yticks(np.arange(0, 1.11, 0.1))
    if np.all(mean_time > 0) and mean_time.max() / mean_time.min() > 100:
        axes[1].set_yscale("log")
    margin = 0.05 * max(float(np.ptp(probabilities)), 0.2)
    for ax, ylabel, panel in zip(
        axes, (r"$\mathbf{F}_1$ score", "Time (s)", r"$\mathbf{F}_1$ score / time"),
        ("(a)", "(b)", "(c)"),
    ):
        ax.set_xlabel(r"Impulse Noise Probability $\mathbf{\pi}$", fontsize=15, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
        ax.set_xticks(probabilities, _probability_labels(probabilities))
        ax.set_xlim(probabilities.min() - margin, probabilities.max() + margin)
        ax.tick_params(axis="both", labelsize=12, direction="in")
        ax.grid(True, linestyle="--", alpha=0.3)
        _panel_label(ax, panel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.51, 1.0),
               ncol=len(methods), prop={"size": 13, "weight": "bold"},
               frameon=True, edgecolor="black", fancybox=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, figure_dir, "signal_metrics")


def _distribution(data, figure_dir: Path) -> None:
    probabilities, methods = data["probabilities"], list(data["methods"])
    labels = (r"$F_1=1$", r"$0.9\leq F_1<1$", r"$0.8\leq F_1<0.9$", r"$F_1<0.8$")
    colors = ("#1F578C", "#329A91", "#E49F3C", "#B43F32")
    trials = data["f1"].shape[0]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.3 * len(methods), 4.4),
                             squeeze=False, sharey=True)
    positions = 1 + np.arange(len(probabilities)) * 0.72
    for k, method in enumerate(methods):
        ax = axes[0, k]
        f1 = data["f1"][:, :, k]
        # Preserve the original exact category boundaries; this is a style change only.
        groups = ((f1 == 1), ((f1 >= 0.9) & (f1 < 1)),
                  ((f1 >= 0.8) & (f1 < 0.9)), (f1 < 0.8))
        bottom = np.zeros(len(probabilities))
        for group, label, color in zip(groups, labels, colors):
            counts = group.sum(0)
            ax.bar(positions, counts, bottom=bottom, width=0.54, color=color,
                   edgecolor="#333333", linewidth=0.35, label=label)
            for x, count, base in zip(positions, counts, bottom):
                if count:
                    small = count < max(2, 0.08 * trials)
                    ax.text(x, base + count / 2, f"{100 * count / trials:.0f}%",
                            ha="center", va="center", fontsize=6.5 if small else 10,
                            color="black" if small else "white", fontweight="bold")
            bottom += counts
        ax.set_xticks(positions, _probability_labels(probabilities))
        ax.set_xlabel(r"Impulse Noise Probability $\mathbf{\pi}$", fontsize=12, fontweight="bold")
        ax.set_ylim(0, trials * 1.06)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.tick_params(axis="both", labelsize=11, direction="in")
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        _panel_label(ax, PAPER_LABELS[str(method)], fontsize=17, y=-0.27)
    axes[0, 0].set_ylabel("Number of Experiments", fontsize=12, fontweight="bold")
    handles = [Patch(facecolor=color, edgecolor="#333333", linewidth=0.35)
               for color in colors]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 1.0),
               ncol=4, fontsize=12, frameon=True, edgecolor="black", fancybox=False)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.26, top=0.84, wspace=0.16)
    save_figure(fig, figure_dir, "signal_f1_distribution", pad_inches=0.03)


def _recovery(data_dir: Path, figure_dir: Path) -> None:
    with np.load(data_dir / "recovery_example.npz", allow_pickle=False) as data:
        truth = data["truth"]
        methods = list(data["methods"])
        rows = len(data["probabilities"])
        fig, axes = plt.subplots(rows, len(methods) + 1, figsize=(12, 2.65 * rows),
                                 squeeze=False, sharex=True, sharey=True)
        x = np.arange(1, len(truth) + 1)
        for row, probability in enumerate(data["probabilities"]):
            axes[row, 0].plot(x, data["observations"][row], color="#808080", linewidth=1.0)
            axes[row, 0].set_title(f"Noisy (Prob={probability:g})", fontsize=11, fontweight="bold")
            for k, method in enumerate(methods):
                ax = axes[row, k + 1]
                ax.plot(x, truth, color="#D62728", linewidth=1.0, label="Original")
                ax.plot(x, data["recovered"][row, k], color="#0072B2", linewidth=1.4, label="Recovery")
                ax.set_title(f"{PAPER_LABELS[str(method)]} (Prob={probability:g})",
                             fontsize=11, fontweight="bold")
                ax.legend(loc="lower left", prop={"size": 8, "weight": "bold"},
                          frameon=True, edgecolor="black", fancybox=False)
            for ax in axes[row]:
                ax.grid(True, linestyle="--", alpha=0.25)
                ax.set_xlim(0, len(truth))
                ax.set_xticks(np.linspace(0, len(truth), 5))
                ax.set_ylim(-15, 15)
                ax.set_yticks(np.arange(-15, 16, 5))
                ax.tick_params(axis="both", labelsize=10, direction="in")
            axes[row, 0].set_ylabel("Amplitude", fontsize=11, fontweight="bold")
        for ax in axes[-1]:
            ax.set_xlabel("Index", fontsize=11, fontweight="bold")
        fig.tight_layout()
        save_figure(fig, figure_dir, "signal_recovery")


def plot(data_dir: Path) -> None:
    configure_plotting()
    data_dir = Path(data_dir)
    figure_dir = data_dir / "figures"
    with np.load(data_dir / "results.npz", allow_pickle=False) as data:
        if not np.all(data["completed"]):
            raise ValueError("Experiment checkpoint is incomplete; finish computation before plotting")
        _comparison(data, figure_dir)
        _distribution(data, figure_dir)
    _recovery(data_dir, figure_dir)
    print(f"Signal figures saved to: {figure_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPOSITORY_ROOT / "results" / "signal")
    args = parser.parse_args()
    plot(args.data)


if __name__ == "__main__":
    main()
