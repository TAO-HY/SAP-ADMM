"""Run the MNIST SAP-ADMM experiment and save its data."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

# Support running this file directly in Spyder or with python /path/to/file.py.
# Resolve imports from this checkout independently of the working directory.
if __package__ in (None, ""):
    import sys
    _project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_project_root))
    sys.path.insert(0, str(_project_root / "src"))
    __package__ = "experiments"

import numpy as np

from sap_admm import sap_admm_image
from sap_admm.utils import compute_gmsd_value, compute_ssim_value, image_operators, psnr

from ._shared import REPOSITORY_ROOT, load_config, save_metadata, save_npz, write_csv


def load_clean_images(digits) -> np.ndarray:
    path = Path(__file__).resolve().parent / "inputs" / "mnist_clean_images.npz"
    with np.load(path, allow_pickle=False) as data:
        indices = [list(data["digits"]).index(digit) for digit in digits]
        return data["images"][indices]


def generate_observation(clean: np.ndarray, config: dict, digit_index: int, trial_index: int):
    seed = config["rng_base"] + 1000 * (digit_index + 1) + trial_index + 1
    rng = np.random.RandomState(seed)
    rows, cols = clean.shape
    mask = rng.rand(rows * cols).reshape(rows, cols, order="F") < config["impulse_prob"]
    replacement = rng.rand(rows * cols).reshape(rows, cols, order="F")
    observation = clean.copy()
    observation[mask] = replacement[mask]
    observation += config["gaussian_std"] * rng.randn(rows * cols).reshape(rows, cols, order="F")
    return np.clip(observation, 0, 1), seed


def run(config: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    save_metadata(output, config, "MNIST total variation denoising")
    clean = load_clean_images(config["digits"])
    R, D = config["trials"], len(config["digits"])
    rows, cols = clean.shape[1:]
    shape = (R, D)
    data = {key: np.full(shape, np.nan) for key in ("psnr", "ssim", "gmsd", "mse", "seconds", "iterations")}
    data.update(
        clean=clean, observations=np.zeros((R, D, rows, cols)),
        recovered=np.zeros((R, D, rows, cols)), seeds=np.zeros(shape, dtype=int),
        nu_final=np.full(shape, np.nan), nu_floor=np.full(shape, np.nan),
        nu_floor_iteration=np.zeros(shape, dtype=int),
        nu_iterations_after_floor=np.zeros(shape, dtype=int),
        nu_post_floor_satisfied=np.zeros(shape, dtype=bool), completed=np.zeros(shape, dtype=bool),
        digits=np.asarray(config["digits"]),
    )
    records = []
    for d, digit in enumerate(config["digits"]):
        operators = image_operators(rows, cols)
        for r in range(R):
            observation, seed = generate_observation(clean[d], config, d, r)
            start = time.perf_counter()
            restored, _, info = sap_admm_image(observation, operators, config["admm"])
            seconds = time.perf_counter() - start
            idx = (r, d)
            values = {
                "psnr": psnr(restored, clean[d]),
                "ssim": compute_ssim_value(restored, clean[d], config["ssim_mode"]),
                "gmsd": compute_gmsd_value(restored, clean[d], config["gmsd_boundary"]),
                "mse": float(np.mean((restored - clean[d]) ** 2)),
                "seconds": seconds, "iterations": info["iter"],
                "nu_final": info["nu_final"], "nu_floor": info["nu_floor"],
                "nu_floor_iteration": info["nu_floor_iteration"],
                "nu_iterations_after_floor": info["nu_iterations_after_floor"],
                "nu_post_floor_satisfied": info["nu_post_floor_satisfied"],
            }
            for key, value in values.items():
                data[key][idx] = value
            data["observations"][idx], data["recovered"][idx] = observation, restored
            data["seeds"][idx], data["completed"][idx] = seed, True
            records.append({
                "Trial": r + 1, "Digit": digit, "PSNR": values["psnr"],
                "SSIM": values["ssim"], "GMSD": values["gmsd"], "MSE": values["mse"],
                "Time_s": seconds, "Iterations": info["iter"],
                "NuFinal": info["nu_final"], "NuFloor": info["nu_floor"],
                "NuFloorIteration": info["nu_floor_iteration"],
                "NuIterationsAfterFloor": info["nu_iterations_after_floor"],
                "NuPostFloorSatisfied": info["nu_post_floor_satisfied"],
                "StopReason": info["stop_reason"],
            })
            print(f"digit={digit} trial={r+1}/{R}: PSNR={values['psnr']:.3f} SSIM={values['ssim']:.4f} GMSD={values['gmsd']:.4f}")
            save_npz(output / "results.npz", **data)
            write_csv(output / "trial_metrics.csv", records)
    summary = []
    for d, digit in enumerate(config["digits"]):
        summary.append({
            "Digit": digit, "Trials": R,
            "MeanPSNR": float(data["psnr"][:, d].mean()),
            "MeanSSIM": float(data["ssim"][:, d].mean()),
            "MeanGMSD": float(data["gmsd"][:, d].mean()),
            "MeanMSE": float(data["mse"][:, d].mean()),
            "MeanTime_s": float(data["seconds"][:, d].mean()),
        })
    write_csv(output / "summary.csv", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPOSITORY_ROOT / "results" / "mnist")
    parser.add_argument("--trials", type=int)
    parser.add_argument("--digits", type=int, nargs="+")
    args = parser.parse_args()
    config = load_config("mnist.json")
    if args.trials is not None:
        if args.trials < 1:
            parser.error("--trials must be positive")
        config["trials"] = args.trials
    if args.digits is not None:
        if len(set(args.digits)) != len(args.digits) or any(d not in range(10) for d in args.digits):
            parser.error("--digits must be unique integers from 0 to 9")
        config["digits"] = args.digits
    run(config, args.out)


if __name__ == "__main__":
    main()
