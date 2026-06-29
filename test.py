"""Plot gravity cosine similarity for the gravity-fit pipeline."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main import compute_gravity_overlays
from src.skeletons.natnet_skeleton import NATNET


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.sum(a * b, axis=-1) / (denom + 1e-12)


def resolve_bones(names: list[str]) -> list[int]:
    indices: list[int] = []
    for name in names:
        if name not in NATNET.names:
            valid = ", ".join(NATNET.names)
            raise ValueError(f"unknown NatNet bone {name!r}; valid bones: {valid}")
        indices.append(NATNET.index(name))
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot gravity cosine similarity for the gravity-fit NatNet path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("parquet_path")
    parser.add_argument(
        "--bone",
        action="append",
        default=None,
        help="NatNet bone to plot. Repeat for multiple bones.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--calibration-start", type=int, default=0)
    parser.add_argument("--calibration-window", type=int, default=600)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_path)
    bone_names = args.bone if args.bone is not None else ["RHand"]
    bone_indices = resolve_bones(bone_names)

    _, _, g_gravity, g_ref, _ = compute_gravity_overlays(
        df,
        calibration_start=args.calibration_start,
        calibration_window=args.calibration_window,
    )

    start = max(args.start, 0)
    end = len(df) if args.end is None else min(args.end, len(df))
    if start >= end:
        raise ValueError("empty frame range")

    frames = np.arange(start, end)
    n_rows = len(bone_indices)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.5 * n_rows), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, bone_idx, bone_name in zip(axes, bone_indices, bone_names):
        cos_gravity = cosine_similarity(g_gravity[:, bone_idx], g_ref[:, bone_idx])

        ax.plot(frames, cos_gravity[start:end], label="gravity-fit NN_R_SS", lw=1.0)
        if args.threshold is not None:
            ax.axhline(args.threshold, color="black", linestyle="--", linewidth=0.8, label="threshold")
        ax.set_ylabel("cosine")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title(bone_name)
        ax.legend(loc="lower right")

    axes[-1].set_xlabel("frame")
    fig.suptitle("Gravity Cosine Similarity: NatNet Path vs SensorSuit Reference", y=0.995)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
