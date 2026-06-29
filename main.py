import argparse

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from src.skeletons.natnet_skeleton import (
    NATNET,
    extract_gravity,
    extract_positions,
    extract_rotations,
)
from src.skeletons.tf import WNN_R_WSS
from src.utils.gravity import available_natnet_to_sensorsuit, estimate_gravity_nn_R_ss
from src.utils.visualization import plot_skeleton


def compute_gravity_overlays(
    df: pd.DataFrame,
    *,
    sensorsuit_suffix: str = "rotation",
    calibration_start: int = 0,
    calibration_window: int = 600,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    """Build skeleton arrays plus NatNet-path and SensorSuit-path gravity vectors.

    ``gravity`` is local IMU gravity rotated through NatNet's bone orientation.
    The intermediate SensorSuit->NatNet transform is fitted from the initial
    T-pose using gravity directions only. ``gravity_ref`` is local
    IMU gravity rotated through the SensorSuit quaternion and then converted
    from SensorSuit world to NatNet world.
    """

    bones_nn = list(NATNET.names)
    pos = extract_positions(df, bones_nn, stream="natnet")
    ori = extract_rotations(df, bones_nn, stream="natnet")
    if ori.shape[1] != len(NATNET):
        raise ValueError("missing one or more NatNet rotation columns")

    n_frames = len(df)
    n_bones = len(NATNET)
    gravity = np.zeros((n_frames, n_bones, 3))
    gravity_ref = np.zeros((n_frames, n_bones, 3))
    mapping = available_natnet_to_sensorsuit(df, sensorsuit_suffix=sensorsuit_suffix)

    for bone_idx, bone_nn in enumerate(bones_nn):
        bone_ss = mapping.get(bone_nn)
        if bone_ss is None:
            continue

        q_ss = extract_rotations(
            df,
            [bone_ss],
            stream="sensorsuit",
            suffix=sensorsuit_suffix,
        )
        g_ss = extract_gravity(df, [bone_ss], stream="sensorsuit")
        if q_ss.shape[1] == 0 or g_ss.shape[1] == 0:
            continue

        wnn_R_nn = Rotation.from_quat(ori[:, bone_idx]).as_matrix()
        wss_R_ss = Rotation.from_quat(q_ss[:, 0]).as_matrix()
        g_ss = g_ss[:, 0]
        nn_R_ss = estimate_gravity_nn_R_ss(
            wnn_R_nn,
            wss_R_ss,
            g_ss,
            start_frame=calibration_start,
            calibration_window=calibration_window,
        )

        g_nn = np.einsum("ij,nj->ni", nn_R_ss, g_ss)
        gravity[:, bone_idx] = np.einsum("nij,nj->ni", wnn_R_nn, g_nn)

        g_wss = np.einsum("nij,nj->ni", wss_R_ss, g_ss)
        gravity_ref[:, bone_idx] = np.einsum("ij,nj->ni", WNN_R_WSS, g_wss)

    return pos, ori, gravity, gravity_ref, mapping


def normalize_frame(frame: int, n_frames: int) -> int:
    if frame < 0:
        frame += n_frames
    if not 0 <= frame < n_frames:
        raise ValueError(f"frame {frame} outside valid range [0, {n_frames - 1}]")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot NatNet skeleton with two SensorSuit gravity overlays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("parquet_path")
    parser.add_argument(
        "--frame",
        type=int,
        action="append",
        help="Frame to plot. Repeat this option to show multiple frames.",
    )
    parser.add_argument("--gravity-len", type=float, default=0.2)
    parser.add_argument("--calibration-start", type=int, default=0)
    parser.add_argument("--calibration-window", type=int, default=600)
    parser.add_argument(
        "--sensorsuit-suffix",
        default="rotation",
        help="SensorSuit quaternion suffix in the parquet columns.",
    )
    parser.add_argument(
        "--print-mapping",
        action="store_true",
        help="Print the NatNet bone -> SensorSuit bone mapping used for gravity overlays.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_path)
    pos, ori, gravity, gravity_ref, mapping = compute_gravity_overlays(
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


if __name__ == "__main__":
    main()
