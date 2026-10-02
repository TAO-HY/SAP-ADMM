"""Read saved general-A results and create the sensitivity figure."""

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
import numpy as np

from ._shared import REPOSITORY_ROOT, configure_plotting, save_figure


def plot(data_dir: Path) -> None:
    configure_plotting()
    with np.load(data_dir / "results.npz", allow_pickle=False) as data:
        if not np.all(data["completed"]):
            raise ValueError("Experiment checkpoint is incomplete; finish computation before plotting")
        x = data["Nmax_list"]
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8))
        for ax, key, label in zip(
            axes, ("f1", "iterations", "mse"),
            (r"Average $\mathbf{F}_1$ score", "Average iterations", "Average MSE")
        ):
            mean = data[key].mean(0)
            ax.plot(x, mean, "o-", color="#0072B2", linewidth=1.6, markersize=5)
            ax.set_xlabel(r"$N_{\max}$", fontsize=12, fontweight="bold")
            ax.set_ylabel(label, fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.tick_params(axis="both", labelsize=10, direction="in")
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.grid(True, linestyle=":", alpha=0.2)
            if np.ptp(mean) < 1e-12:
                padding = max(abs(float(mean[0])) * 0.01, 1e-4)
                ax.set_ylim(float(mean[0]) - padding, float(mean[0]) + padding)
        fig.tight_layout()
        save_figure(fig, data_dir / "figures", "general_A_sensitivity")
    print(f"General-A figures saved to: {data_dir / 'figures'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPOSITORY_ROOT / "results" / "general_a")
    args = parser.parse_args()
    plot(args.data)


if __name__ == "__main__":
    main()
