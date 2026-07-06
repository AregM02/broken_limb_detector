"""Single CLI entry point for detection and gravity diagnostics."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from src.detector.checkers import AbsoluteLimitChecker, BaseChecker, GravityAlignmentChecker
from src.detector.detector import detect_breaks, format_report
from src.skeletons.natnet_skeleton import NATNET, extract_positions, extract_rotations
from src.utils.gravity import compute_gravity_vectors, cosine_similarity
from src.utils.transforms import global_to_local
from src.utils.visualization import plot_bone_with_violations, plot_skeleton


def load_gravity_overlays(
    df: pd.DataFrame,
    *,
    sensorsuit_suffix: str = "auto",
    calibration_start: int = 0,
    calibration_window: int = 600,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    """Load NatNet pose arrays and fitted gravity overlay vectors."""

    bones = list(NATNET.names)
    pos = extract_positions(df, bones, stream="natnet")
    ori = extract_rotations(df, bones, stream="natnet")
    if ori.shape[1] != len(NATNET):
        raise ValueError("missing one or more NatNet rotation columns")

    gravity, gravity_ref, mapping = compute_gravity_vectors(
        df,
        sensorsuit_suffix=sensorsuit_suffix,
        calibration_start=calibration_start,
        calibration_window=calibration_window,
        natnet_rotations=ori,
    )
    return pos, ori, gravity, gravity_ref, mapping


def normalize_frame(frame: int, n_frames: int) -> int:
    if frame < 0:
        frame += n_frames
    if not 0 <= frame < n_frames:
        raise ValueError(f"frame {frame} outside valid range [0, {n_frames - 1}]")
    return frame


def resolve_bones(names: list[str]) -> list[int]:
    indices: list[int] = []
    for name in names:
        if name not in NATNET.names:
            valid = ", ".join(NATNET.names)
            raise ValueError(f"unknown NatNet bone {name!r}; valid bones: {valid}")
        indices.append(NATNET.index(name))
    return indices


def run_skeleton(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.parquet_path)
    pos, ori, gravity, gravity_ref, mapping = load_gravity_overlays(
        df,
        sensorsuit_suffix=args.sensorsuit_suffix,
        calibration_start=args.calibration_start,
        calibration_window=args.calibration_window,
    )

    if args.print_mapping:
        for bone_nn in NATNET.names:
            bone_ss = mapping.get(bone_nn)
            if bone_ss is not None:
                print(f"{bone_nn:>10s} <- {bone_ss}")

    frames = args.frame if args.frame is not None else [min(1000, len(df) - 1)]
    for frame in frames:
        plot_skeleton(
            pos,
            ori,
            frame=normalize_frame(frame, len(df)),
            gravity=gravity,
            gravity_ref=gravity_ref,
            gravity_len=args.gravity_len,
        )


def run_cosine(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.parquet_path)
    bone_names = args.bone if args.bone is not None else ["RHand"]
    bone_indices = resolve_bones(bone_names)
    _, _, gravity, gravity_ref, _ = load_gravity_overlays(
        df,
        sensorsuit_suffix=args.sensorsuit_suffix,
        calibration_start=args.calibration_start,
        calibration_window=args.calibration_window,
    )

    start = max(args.start, 0)
    end = len(df) if args.end is None else min(args.end, len(df))
    if start >= end:
        raise ValueError("empty frame range")

    frames = np.arange(start, end)
    fig, axes = plt.subplots(len(bone_indices), 1, figsize=(14, 3.5 * len(bone_indices)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, bone_idx, bone_name in zip(axes, bone_indices, bone_names):
        cos_gravity = cosine_similarity(gravity[:, bone_idx], gravity_ref[:, bone_idx])
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


def run_detect(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.parquet_path)
    checkers: list[BaseChecker] = [
        AbsoluteLimitChecker(calibrate=args.calibrate, tpose_window=args.tpose_window),
        # GravityAlignmentChecker(
        #     threshold_cos=args.gravity_threshold,
        #     calibration_start=args.calibration_start,
        #     calibration_window=args.calibration_window,
        #     sensorsuit_suffix=args.sensorsuit_suffix,
        # ),
    ]

    result = detect_breaks(df, checkers=checkers)
    print(format_report(result, max_rows=args.max_rows))

    if not args.plot:
        return

    rpy_result = result.checker_results.get("AbsoluteLimitChecker")
    gravity_result = result.checker_results.get("GravityAlignmentChecker")
    if rpy_result is None and gravity_result is None:
        return

    gravity_violated = None if gravity_result is None else gravity_result.violated
    if rpy_result is None:
        quats = extract_rotations(df, list(NATNET.names))
        pos = extract_positions(df, list(NATNET.names))
        local_quats, _ = global_to_local(quats, pos)
        rpy = Rotation.from_quat(local_quats).as_euler("xyz", degrees=True)
        plot_items = [(bone_name, None) for bone_name in gravity_result.details["bones"]]
        violated_axes = None
    else:
        rpy = result.rpy
        plot_items = rpy_result.details["limits"].items()
        violated_axes = result.violated_axes

    for bone_name, limits in plot_items:
        plot_bone_with_violations(
            bone_name,
            violated=result.violated,
            violated_axes=violated_axes,
            limits=limits,
            rpy=rpy,
            gravity_violated=gravity_violated,
            plot_broken_edges=args.plot_broken_edges,
        )


def add_gravity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--calibration-start", type=int, default=0)
    parser.add_argument("--calibration-window", type=int, default=600)
    parser.add_argument(
        "--sensorsuit-suffix",
        default="auto",
        help="SensorSuit quaternion suffix in the parquet columns: auto, orientation, or rotation.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Broken-limb detector and gravity diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="Run the break detector.")
    detect.add_argument("parquet_path")
    detect.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=False)
    detect.add_argument("--tpose-window", type=int, default=600)
    detect.add_argument("--gravity-threshold", type=float, default=0.8)
    detect.add_argument("--max-rows", type=int, default=20)
    detect.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    detect.add_argument(
        "--plot-broken-edges",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw debounced broken_start/broken_end vertical markers.",
    )
    add_gravity_args(detect)
    detect.set_defaults(handler=run_detect)

    skeleton = subparsers.add_parser("skeleton", help="Plot skeleton with gravity overlays.")
    skeleton.add_argument("parquet_path")
    skeleton.add_argument("--frame", type=int, action="append", help="Frame to plot; repeat for multiple frames.")
    skeleton.add_argument("--gravity-len", type=float, default=0.2)
    skeleton.add_argument("--print-mapping", action="store_true")
    add_gravity_args(skeleton)
    skeleton.set_defaults(handler=run_skeleton)

    cosine = subparsers.add_parser("cosine", help="Plot gravity cosine similarity.")
    cosine.add_argument("parquet_path")
    cosine.add_argument("--bone", action="append", default=None, help="NatNet bone to plot; repeat for more.")
    cosine.add_argument("--start", type=int, default=0)
    cosine.add_argument("--end", type=int, default=None)
    cosine.add_argument("--threshold", type=float, default=None)
    add_gravity_args(cosine)
    cosine.set_defaults(handler=run_cosine)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
