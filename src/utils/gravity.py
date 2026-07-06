"""Gravity-direction calibration helpers."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from src.skeletons.natnet_skeleton import (
    NATNET,
    SENSORSUIT_TO_NATNET,
    extract_gravity,
    extract_rotations,
)
from src.skeletons.tf import WNN_R_WSS


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / (norm + 1e-12)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return per-vector cosine similarity along the last axis."""

    return np.sum(normalize_vectors(a) * normalize_vectors(b), axis=-1)


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


def compute_gravity_vectors(
    df,
    *,
    sensorsuit_suffix: str = "rotation",
    calibration_start: int = 0,
    calibration_window: int = 600,
    natnet_rotations: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Compute world-frame gravity vectors for the gravity-fit pipeline.

    Returns ``gravity`` from the NatNet path, ``gravity_ref`` from the
    SensorSuit orientation path, and the NatNet bone -> SensorSuit bone mapping
    used for bones with all required columns present.
    """

    bones_nn = list(NATNET.names)
    if natnet_rotations is None:
        natnet_rotations = extract_rotations(df, bones_nn, stream="natnet")
    if natnet_rotations.shape[1] != len(NATNET):
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

        wnn_R_nn = Rotation.from_quat(natnet_rotations[:, bone_idx]).as_matrix()
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

    return gravity, gravity_ref, mapping
