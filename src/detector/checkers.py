"""Validation checkers used by the break detector."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from src.skeletons.natnet_skeleton import NATNET, extract_positions, extract_rotations
from src.utils.gravity import compute_gravity_vectors, cosine_similarity
from src.utils.transforms import global_to_local


ABSOLUTE_LIMITS: dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {
    # "RHand": ((-45, 45), (-45, 40), (-105, 80)),
    "RHand": ((-60, 50), (-50, 40), (-105, 80)),
    "LHand": ((-50, 60), (-40, 50), (-80, 105)),
    "RFoot": ((-25, 50), (-90, 40), (-60, 40)),
    "LFoot": ((-25, 50), (-40, 90), (-40, 60)),
}
END_EFFECTOR_BONES: tuple[str, ...] = tuple(ABSOLUTE_LIMITS)


class CheckerResult(NamedTuple):
    """Result payload from one checker."""

    violated: np.ndarray
    details: dict[str, Any]


class BaseChecker(ABC):
    """Base class for detector validation checks."""

    @abstractmethod
    def check(self, df: pd.DataFrame) -> CheckerResult:
        """Run the checker on a tracking DataFrame."""


class AbsoluteLimitChecker(BaseChecker):
    """Validate local joint angles against absolute RPY limits."""

    def __init__(
        self,
        limits: dict[str | int, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] | None = None,
        calibrate: bool = False,
        tpose_window: int = 600,
    ):
        self.limits = limits if limits is not None else ABSOLUTE_LIMITS
        self.calibrate = calibrate
        self.tpose_window = tpose_window

    def check(self, df: pd.DataFrame) -> CheckerResult:
        quats = extract_rotations(df, list(NATNET.names))
        pos = extract_positions(df, list(NATNET.names))
        local_quats, _ = global_to_local(quats, pos)

        if self.calibrate and self.tpose_window > 0:
            n = min(self.tpose_window, local_quats.shape[0])
            q_offset = local_quats[:n].mean(axis=0)
            local_quats = (Rotation.from_quat(q_offset).inv() * Rotation.from_quat(local_quats)).as_quat()

        rpy = Rotation.from_quat(local_quats).as_euler("xyz", degrees=True)
        n_frames, n_bones, _ = rpy.shape
        violated = np.zeros((n_frames, n_bones), dtype=bool)
        violated_axes = np.zeros((n_frames, n_bones, 3), dtype=bool)

        for key, axes in self.limits.items():
            bone_idx = NATNET.index(key) if isinstance(key, str) else key
            if bone_idx >= n_bones:
                continue

            (r_lo, r_hi), (p_lo, p_hi), (y_lo, y_hi) = axes
            bone_rpy = rpy[:, bone_idx, :]
            axis_violations = np.stack(
                [
                    (bone_rpy[:, 0] < r_lo) | (bone_rpy[:, 0] > r_hi),
                    (bone_rpy[:, 1] < p_lo) | (bone_rpy[:, 1] > p_hi),
                    (bone_rpy[:, 2] < y_lo) | (bone_rpy[:, 2] > y_hi),
                ],
                axis=1,
            )

            violated_axes[:, bone_idx] = axis_violations
            violated[:, bone_idx] = axis_violations.any(axis=1)

        return CheckerResult(
            violated=violated,
            details={"violated_axes": violated_axes, "rpy": rpy, "limits": self.limits},
        )


class GravityAlignmentChecker(BaseChecker):
    """Compare the gravity-fit NatNet path against the SensorSuit gravity path."""

    def __init__(
        self,
        threshold_cos: float = 0.8,
        calibration_start: int = 0,
        calibration_window: int = 600,
        sensorsuit_suffix: str = "auto",
        bones: tuple[str, ...] | list[str] | None = None,
    ):
        self.threshold_cos = threshold_cos
        self.calibration_start = calibration_start
        self.calibration_window = calibration_window
        self.sensorsuit_suffix = sensorsuit_suffix
        self.bones = END_EFFECTOR_BONES if bones is None else tuple(bones)
        for bone in self.bones:
            if bone not in NATNET.names:
                raise ValueError(f"unknown NatNet bone {bone!r}")

    def check(self, df: pd.DataFrame) -> CheckerResult:
        gravity, gravity_ref, mapping = compute_gravity_vectors(
            df,
            sensorsuit_suffix=self.sensorsuit_suffix,
            calibration_start=self.calibration_start,
            calibration_window=self.calibration_window,
            bone_names=self.bones,
        )
        n_frames, n_bones, _ = gravity.shape
        violated = np.zeros((n_frames, n_bones), dtype=bool)
        angular_errors = np.full((n_frames, n_bones), np.nan)
        cosine_scores = np.full((n_frames, n_bones), np.nan)

        valid = (
            (np.linalg.norm(gravity, axis=2) > 1e-8)
            & (np.linalg.norm(gravity_ref, axis=2) > 1e-8)
        )
        cos_theta = np.clip(cosine_similarity(gravity, gravity_ref), -1.0, 1.0)

        cosine_scores[valid] = cos_theta[valid]
        angular_errors[valid] = np.degrees(np.arccos(cos_theta[valid]))
        violated[valid] = cos_theta[valid] < self.threshold_cos

        return CheckerResult(
            violated=violated,
            details={
                "angular_errors": angular_errors,
                "cosine_similarity": cosine_scores,
                "mapping": mapping,
                "valid": valid,
                "bones": self.bones,
                "threshold_cos": self.threshold_cos,
            },
        )
