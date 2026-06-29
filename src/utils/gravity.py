"""Gravity-direction calibration helpers."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from src.skeletons.natnet_skeleton import SENSORSUIT_TO_NATNET
from src.skeletons.tf import WNN_R_WSS


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / (norm + 1e-12)


def available_natnet_to_sensorsuit(
    df,
    *,
    sensorsuit_suffix: str = "rotation",
) -> dict[str, str]:
    """Map NatNet bones to SensorSuit bones with required columns present."""

    mapping: dict[str, str] = {}
    for bone_ss, bone_nn in SENSORSUIT_TO_NATNET.items():
        has_nn = all(f"natnet_{bone_nn}_rotation_{axis}" in df.columns for axis in "xyzw")
        has_ss = all(
            f"sensorsuit_{bone_ss}_{sensorsuit_suffix}_{axis}" in df.columns
            for axis in "xyzw"
        )
        has_gravity = all(f"sensorsuit_{bone_ss}_gravity_{axis}" in df.columns for axis in "xyz")
        if has_nn and has_ss and has_gravity and bone_nn not in mapping:
            mapping[bone_nn] = bone_ss
    return mapping


def estimate_gravity_nn_R_ss(
    wnn_R_nn: np.ndarray,
    wss_R_ss: np.ndarray,
    g_ss: np.ndarray,
    *,
    start_frame: int = 0,
    calibration_window: int = 600,
    wnn_R_wss: np.ndarray = WNN_R_WSS,
) -> np.ndarray:
    """Fit ``NN_R_SS`` from gravity directions in a calibration window.

    This estimates only the transform needed to make local gravity agree
    between the NatNet path and the SensorSuit path.  It is not a full rigid
    sensor-to-bone orientation calibration.
    """

    stop_frame = min(len(g_ss), start_frame + calibration_window)
    if start_frame < 0 or calibration_window <= 0 or start_frame >= stop_frame:
        raise ValueError("invalid gravity calibration window")

    g_wss = np.einsum("nij,nj->ni", wss_R_ss[start_frame:stop_frame], g_ss[start_frame:stop_frame])
    g_ref = np.einsum("ij,nj->ni", wnn_R_wss, g_wss)
    g_target_nn = np.einsum("nji,nj->ni", wnn_R_nn[start_frame:stop_frame], g_ref)

    rotation, _ = Rotation.align_vectors(
        normalize_vectors(g_target_nn),
        normalize_vectors(g_ss[start_frame:stop_frame]),
    )
    return rotation.as_matrix()
