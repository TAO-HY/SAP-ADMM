"""Shared experiment helpers."""

from __future__ import annotations

import io
import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import scipy
import matplotlib
from threadpoolctl import threadpool_info

from sap_admm.utils import save_npz, write_csv

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent / "config"


def load_config(name: str) -> dict:
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


def save_metadata(folder: Path, config: dict, experiment: str) -> None:
    payload = {
        "experiment": experiment,
        "config": config,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "blas": threadpool_info(),
        "timer": "time.perf_counter (elapsed wall time)",
        "scope": "proposed SAP-ADMM algorithms only",
        "random_generator": "NumPy RandomState (MT19937)",
    }
    (folder / "metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def save_figure(fig, folder: Path, basename: str, *, pad_inches: float = 0.1) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {})):
        buffer = io.BytesIO()
        fig.savefig(buffer, format=suffix, bbox_inches="tight",
                    pad_inches=pad_inches, facecolor="white", **kwargs)
        (folder / f"{basename}.{suffix}").write_bytes(buffer.getvalue())
    plt.close(fig)


def configure_plotting() -> None:
    """Use the reference figures' Times-style typography with portable fallbacks."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next((name for name in (
        "Times New Roman", "Nimbus Roman", "Liberation Serif", "STIXGeneral"
    ) if name in available), "DejaVu Serif")
    plt.rcParams.update({
        "font.family": family,
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        "mathtext.sf": family,
        "mathtext.tt": family,
        "mathtext.cal": "STIXGeneral",
        "mathtext.fallback": "stix",
        "axes.unicode_minus": False,
        "axes.linewidth": 1.0,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


__all__ = [
    "REPOSITORY_ROOT", "configure_plotting", "load_config", "save_figure",
    "save_metadata", "save_npz", "write_csv",
]
