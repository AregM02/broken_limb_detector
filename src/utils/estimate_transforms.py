"""Estimate gravity-specific SensorSuit -> NatNet local transforms.

This utility intentionally does not estimate a full rigid sensor-to-bone
orientation.  It estimates the rotation used by the gravity pipeline only: the
rotation that maps local SensorSuit gravity into the NatNet bone frame so that
NatNet and SensorSuit gravity agree during a calibration window.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from src.skeletons.natnet_skeleton import SENSORSUIT_TO_NATNET, extract_gravity, extract_rotations
from src.skeletons.tf import WNN_R_WSS
from src.utils.gravity import (
    available_natnet_to_sensorsuit,
    estimate_gravity_nn_R_ss,
    normalize_vectors,
    resolve_sensorsuit_suffix,
)


@dataclass(frozen=True)
class GravityTransformEstimate:
    bone_ss: str
    bone_nn: str
    matrix: np.ndarray
    n_frames: int
    mean_error_deg: float
    median_error_deg: float
    p95_error_deg: float
    max_error_deg: float


def world_transform_with_yaw(world_yaw_deg: float = 0.0) -> np.ndarray:
    """Return WNN_R_WSS with optional yaw about NatNet's +Y up axis."""

    yaw = Rotation.from_euler("y", world_yaw_deg, degrees=True).as_matrix()
    return yaw @ WNN_R_WSS


def _angular_error_deg(
    wnn_R_nn: np.ndarray,
    wss_R_ss: np.ndarray,
    g_ss: np.ndarray,
    nn_R_ss: np.ndarray,
    *,
    start_frame: int,
    calibration_window: int,
    wnn_R_wss: np.ndarray,
) -> np.ndarray:
    stop_frame = min(len(g_ss), start_frame + calibration_window)
    g_pred_nn = np.einsum("ij,nj->ni", nn_R_ss, g_ss[start_frame:stop_frame])
    g_wss = np.einsum("nij,nj->ni", wss_R_ss[start_frame:stop_frame], g_ss[start_frame:stop_frame])
    g_ref = np.einsum("ij,nj->ni", wnn_R_wss, g_wss)
    g_target_nn = np.einsum("nji,nj->ni", wnn_R_nn[start_frame:stop_frame], g_ref)
    cos_theta = np.sum(normalize_vectors(g_pred_nn) * normalize_vectors(g_target_nn), axis=1)
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def estimate_one_gravity_transform(
    df: pd.DataFrame,
    bone_ss: str,
    *,
    bone_nn: str | None = None,
    start_frame: int = 0,
    calibration_window: int = 600,
    wnn_R_wss: np.ndarray | None = None,
    sensorsuit_suffix: str = "auto",
) -> GravityTransformEstimate:
    if bone_nn is None:
        bone_nn = SENSORSUIT_TO_NATNET[bone_ss]
    if wnn_R_wss is None:
        wnn_R_wss = WNN_R_WSS
    sensorsuit_suffix = resolve_sensorsuit_suffix(df, sensorsuit_suffix)

    q_nn = extract_rotations(df, [bone_nn], stream="natnet")
    q_ss = extract_rotations(df, [bone_ss], stream="sensorsuit", suffix=sensorsuit_suffix)
    g_ss = extract_gravity(df, [bone_ss], stream="sensorsuit")
    if q_nn.shape[1] == 0:
        raise KeyError(f"missing NatNet rotation columns for {bone_nn!r}")
    if q_ss.shape[1] == 0:
        raise KeyError(f"missing SensorSuit rotation columns for {bone_ss!r}")
    if g_ss.shape[1] == 0:
        raise KeyError(f"missing SensorSuit gravity columns for {bone_ss!r}")

    wnn_R_nn = Rotation.from_quat(q_nn[:, 0]).as_matrix()
    wss_R_ss = Rotation.from_quat(q_ss[:, 0]).as_matrix()
    g_ss = g_ss[:, 0]
    matrix = estimate_gravity_nn_R_ss(
        wnn_R_nn,
        wss_R_ss,
        g_ss,
        start_frame=start_frame,
        calibration_window=calibration_window,
        wnn_R_wss=wnn_R_wss,
    )
    error_deg = _angular_error_deg(
        wnn_R_nn,
        wss_R_ss,
        g_ss,
        matrix,
        start_frame=start_frame,
        calibration_window=calibration_window,
        wnn_R_wss=wnn_R_wss,
    )

    return GravityTransformEstimate(
        bone_ss=bone_ss,
        bone_nn=bone_nn,
        matrix=matrix,
        n_frames=len(error_deg),
        mean_error_deg=float(np.mean(error_deg)),
        median_error_deg=float(np.median(error_deg)),
        p95_error_deg=float(np.percentile(error_deg, 95)),
        max_error_deg=float(np.max(error_deg)),
    )


def estimate_gravity_transforms(
    df: pd.DataFrame,
    bones_ss: Sequence[str] | None = None,
    *,
    start_frame: int = 0,
    calibration_window: int = 600,
    wnn_R_wss: np.ndarray | None = None,
    sensorsuit_suffix: str = "auto",
) -> dict[str, GravityTransformEstimate]:
    if bones_ss is None:
        mapping = available_natnet_to_sensorsuit(df, sensorsuit_suffix=sensorsuit_suffix)
        bones_ss = tuple(mapping.values())

    estimates: dict[str, GravityTransformEstimate] = {}
    for bone_ss in bones_ss:
        estimates[bone_ss] = estimate_one_gravity_transform(
            df,
            bone_ss,
            start_frame=start_frame,
            calibration_window=calibration_window,
            wnn_R_wss=wnn_R_wss,
            sensorsuit_suffix=sensorsuit_suffix,
        )
    return estimates


def _format_matrix(matrix: np.ndarray, indent: str = "        ") -> str:
    rows = ["[" + ", ".join(f"{value: .8f}" for value in row) + "]" for row in matrix]
    return "np.array([" + (",\n" + indent).join(rows) + "])"


def _print_estimates(estimates: dict[str, GravityTransformEstimate], wnn_R_wss: np.ndarray) -> None:
    print("WNN_R_WSS = " + _format_matrix(wnn_R_wss, indent="             "))
    print()
    print("GRAVITY_TRANSFORMS = {")
    for bone_ss, estimate in estimates.items():
        diag = (
            f"# {estimate.bone_nn}, n={estimate.n_frames}, "
            f"mean={estimate.mean_error_deg:.2f} deg, "
            f"p95={estimate.p95_error_deg:.2f} deg"
        )
        print(f"    {bone_ss!r}: {_format_matrix(estimate.matrix)},  {diag}")
    print("}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate gravity-specific SensorSuit -> NatNet local-frame rotations."
    )
    parser.add_argument("parquet_path")
    parser.add_argument("--bone", dest="bones", action="append", help="SensorSuit bone to estimate.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--calibration-window", "--tpose-window", type=int, default=600)
    parser.add_argument(
        "--world-yaw-deg",
        type=float,
        default=0.0,
        help="Extra yaw around NatNet +Y applied to WNN_R_WSS.",
    )
    parser.add_argument(
        "--sensorsuit-suffix",
        default="auto",
        help="SensorSuit quaternion suffix: auto, orientation, or rotation.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_path)
    wnn_R_wss = world_transform_with_yaw(args.world_yaw_deg)
    estimates = estimate_gravity_transforms(
        df,
        args.bones,
        start_frame=args.start_frame,
        calibration_window=args.calibration_window,
        wnn_R_wss=wnn_R_wss,
        sensorsuit_suffix=args.sensorsuit_suffix,
    )
    _print_estimates(estimates, wnn_R_wss)


if __name__ == "__main__":
    main()
