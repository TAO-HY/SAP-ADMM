"""Read saved MNIST results and create the SAP-ADMM image grid."""

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
        count = len(data["digits"])
        if count == 0:
            raise ValueError("No MNIST digits are available to plot")
        # Keep physical tiles square and gaps small, including for digit subsets.
        width = min(10.0, 0.93 * count + 0.7)
        left, right, gap_x = 0.55 / width, 0.15 / width, 0.03 / width
        top, bottom, gap_y = 0.025, 0.020, 0.003
        tile_width = (1 - left - right - (count - 1) * gap_x) / count
        tile_height = (1 - top - bottom - 2 * gap_y) / 3
        height = width * tile_width / tile_height
        fig = plt.figure(figsize=(width, height), facecolor="white")
        arrays = (data["clean"], data["observations"][0], data["recovered"][0])
        labels = ("Original", "Noisy", "SAP-ADMM")
        for row, (images, label) in enumerate(zip(arrays, labels)):
            for col in range(count):
                ax = fig.add_axes([
                    left + col * (tile_width + gap_x),
                    1 - top - (row + 1) * tile_height - row * gap_y,
                    tile_width, tile_height,
                ])
                ax.imshow(images[col], cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest", aspect="equal")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if col == 0:
                    ax.set_ylabel(label, fontsize=12, fontweight="bold", labelpad=10)
        save_figure(fig, data_dir / "figures", "mnist_restoration", pad_inches=0.04)
    print(f"MNIST figures saved to: {data_dir / 'figures'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPOSITORY_ROOT / "results" / "mnist")
    args = parser.parse_args()
    plot(args.data)


if __name__ == "__main__":
    main()
