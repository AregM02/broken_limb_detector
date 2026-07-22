"""Validation checkers used by the break detector."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from src.skeletons.natnet_skeleton import NATNET, extract_positions, extract_rotations
from src.utils.gravity import compute_gravity_vectors, cosine_similarity
from src.utils.transforms import global_to_local
from src.utils.visualization import (
    plot_angular_velocity,
    plot_linear_acceleration,
    plot_violations,
)

# Extrinsic rotations around fixed parent frame XYZ axes
ABSOLUTE_LIMITS: dict[str, np.ndarray] = {
    "RHand": np.array([[-45, 50], [-33, 36], [-88, 63]], dtype=float),
    "LHand": np.array([[-50, 43], [-36, 33], [-63, 88]], dtype=float),
    "RFoot": np.array([[-22, 44], [-73, 48.5], [-85, 66]], dtype=float),
    "LFoot": np.array([[-15, 52], [-43, 73], [-23, 43]], dtype=float),
}
END_EFFECTOR_BONES: tuple[str, ...] = tuple(ABSOLUTE_LIMITS)

# Maximum calibrated IMU acceleration magnitude in validation.parquet [m/s^2].
# Ab and Neck use the nearest available IMUs: lower_back and head.
LINEAR_ACCELERATION_LIMITS: dict[str, float] = {
    "Hip": 29.583,
    "Ab": 29.583,
    "Chest": 17.942,
    "Neck": 21.024,
    "LShoulder": 27.288,
    "RShoulder": 21.575,
    "LUArm": 53.977,
    "RUArm": 45.548,
    "LFArm": 87.811,
    "RFArm": 110.759,
    "LHand": 106.266,
    "RHand": 112.624,
    "LThigh": 28.769,
    "RThigh": 50.415,
    "LShin": 41.114,
    "RShin": 63.353,
    "LFoot": 38.966,
    "RFoot": 76.364,
    "Head": 21.024,
}


class BaseChecker(ABC):
    """Base interface for checkers that return one boolean value per frame and bone."""

    def check(self, df: pd.DataFrame) -> np.ndarray:
        """Run the checker and validate its full-skeleton boolean mask."""

        mask = self._check(df)
        expected = (len(df), len(NATNET))
        if mask.shape != expected:
            raise ValueError(f"{self.__class__.__name__} returned mask shape {mask.shape}, expected {expected}")
        if mask.dtype != bool:
            raise TypeError(f"{self.__class__.__name__} must return a boolean mask")
        return mask

    @abstractmethod
    def _check(self, df: pd.DataFrame) -> np.ndarray:
        """Implement the checker-specific classification logic."""

        raise NotImplementedError

    @abstractmethod
    def diagnostics(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return checker-specific continuous values and classifications."""

        raise NotImplementedError


class AbsoluteLimitChecker(BaseChecker):
    """Validate local joint angles against absolute RPY limits."""

    def __init__(
        self,
        limits: dict[str | int, np.ndarray] | None = None,
        margins: dict[str | int, np.ndarray] | None = None,
        calibrate: bool = False,
        tpose_window: int = 600,
        plot: bool = False,
    ):
        self.limits = limits if limits is not None else ABSOLUTE_LIMITS
        self.margins = margins
        self.calibrate = calibrate
        self.tpose_window = tpose_window
        self.plot = plot
        self._cached_df: pd.DataFrame | None = None
        self._cached_calibration: tuple[bool, int] | None = None
        self._cached_local_quats: np.ndarray | None = None
        self._cached_rpy: np.ndarray | None = None

    def clear_cache(self) -> None:
        self._cached_df = None
        self._cached_calibration = None
        self._cached_local_quats = None
        self._cached_rpy = None

    def _effective_bounds(self, key: str | int) -> np.ndarray:
        bounds = np.array(self.limits[key], dtype=float, copy=True)
        if bounds.shape != (3, 2):
            raise ValueError(f"limits for {key!r} must have shape (3, 2)")
        if self.margins is None or key not in self.margins:
            return bounds

        margins = np.asarray(self.margins[key], dtype=float)
        if margins.shape != (3, 2):
            raise ValueError(f"margins for {key!r} must have shape (3, 2)")

        bounds[:, 0] -= margins[:, 0]
        bounds[:, 1] += margins[:, 1]
        return bounds

    def _get_quats_and_rpy(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # Data has been cached once before
        calibration = (self.calibrate, self.tpose_window)
        if self._cached_df is df and self._cached_calibration == calibration:
            return self._cached_local_quats, self._cached_rpy

        quats = extract_rotations(df, list(NATNET.names))
        if quats.shape[1] != len(NATNET):
            raise ValueError("missing one or more NatNet rotation columns")
        local_quats = global_to_local(quats)

        if self.calibrate and self.tpose_window > 0:
            n = min(self.tpose_window, local_quats.shape[0])
            q_offset = local_quats[:n].mean(axis=0)
            local_quats = (Rotation.from_quat(q_offset).inv() * Rotation.from_quat(local_quats)).as_quat()

        rpy = Rotation.from_quat(local_quats).as_euler("xyz", degrees=True)
        self._cached_df = df
        self._cached_calibration = calibration
        self._cached_local_quats = local_quats
        self._cached_rpy = rpy
        return local_quats, rpy

    def diagnostics(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return calibrated orientations, RPY values, limits, and excesses."""

        local_quats, rpy = self._get_quats_and_rpy(df)
        n_frames, n_bones, _ = rpy.shape
        violated = np.zeros((n_frames, n_bones), dtype=bool)
        axis_excess = np.full((n_frames, n_bones, 3), np.nan)
        effective_limits = np.full((n_bones, 3, 2), np.nan)

        for key in self.limits:
            bone_idx = NATNET.index(key) if isinstance(key, str) else key
            if bone_idx >= n_bones:
                continue

            bounds = self._effective_bounds(key)
            bone_rpy = rpy[:, bone_idx, :]
            excess = np.maximum(bounds[:, 0] - bone_rpy, bone_rpy - bounds[:, 1])
            axis_excess[:, bone_idx] = excess
            effective_limits[bone_idx] = bounds
            violated[:, bone_idx] = (excess > 0).any(axis=1)

        return {
            "violated": violated,
            "local_quaternions": local_quats,
            "rpy": rpy,
            "axis_excess": axis_excess,
            "effective_limits": effective_limits,
        }

    def _check(self, df: pd.DataFrame) -> np.ndarray:
        details = self.diagnostics(df)

        if self.plot:
            for key in self.limits:
                bone_idx = NATNET.index(key) if isinstance(key, str) else key
                if bone_idx >= len(NATNET):
                    continue
                plot_violations(
                    NATNET[bone_idx],
                    details["local_quaternions"],
                    details["violated"][:, bone_idx],
                    details["effective_limits"][bone_idx],
                )

        return details["violated"]


class GravityAlignmentChecker(BaseChecker):
    """Compare the gravity-fit NatNet path against constant NatNet-world gravity."""

    def __init__(
        self,
        threshold_cos: float = 0.8,
        calibration_start: int = 0,
        calibration_window: int = 600,
        bones: tuple[str, ...] | list[str] | None = None,
        temporal_filter: bool = False,
        temporal_window: int = 40,
        temporal_required: int = 15,
    ):
        self.threshold_cos = threshold_cos
        self.calibration_start = calibration_start
        self.calibration_window = calibration_window
        self.bones = END_EFFECTOR_BONES if bones is None else tuple(bones)
        self.temporal_filter = temporal_filter
        self.temporal_window = temporal_window
        self.temporal_required = temporal_required
        if not 1 <= temporal_required <= temporal_window:
            raise ValueError("temporal_required must be between 1 and temporal_window")
        for bone in self.bones:
            if bone not in NATNET.names:
                raise ValueError(f"unknown NatNet bone {bone!r}")
        self._cached_df: pd.DataFrame | None = None
        self._cached_config: tuple[int, int, tuple[str, ...]] | None = None
        self._cached_gravity: np.ndarray | None = None
        self._cached_gravity_ref: np.ndarray | None = None
        self._cached_valid: np.ndarray | None = None
        self._cached_cosine: np.ndarray | None = None

    def clear_cache(self) -> None:
        self._cached_df = None
        self._cached_config = None
        self._cached_gravity = None
        self._cached_gravity_ref = None
        self._cached_valid = None
        self._cached_cosine = None

    def _get_gravity_info(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        config = (self.calibration_start, self.calibration_window, self.bones)
        if self._cached_df is df and self._cached_config == config:
            return (
                self._cached_gravity,
                self._cached_gravity_ref,
                self._cached_valid,
                self._cached_cosine,
            )

        gravity, gravity_ref, _ = compute_gravity_vectors(
            df,
            calibration_start=self.calibration_start,
            calibration_window=self.calibration_window,
            bone_names=self.bones,
        )

        valid = (
            (np.linalg.norm(gravity, axis=2) > 1e-8)
            & (np.linalg.norm(gravity_ref, axis=2) > 1e-8)
        )
        cos_theta = np.clip(cosine_similarity(gravity, gravity_ref), -1.0, 1.0)

        self._cached_df = df
        self._cached_config = config
        self._cached_gravity = gravity
        self._cached_gravity_ref = gravity_ref
        self._cached_valid = valid
        self._cached_cosine = cos_theta
        return gravity, gravity_ref, valid, cos_theta

    def _confirm_over_time(self, mask: np.ndarray) -> np.ndarray:
        kernel = np.ones(self.temporal_window, dtype=int)
        counts = np.convolve(mask.astype(int), kernel, mode="full")[:len(mask)]
        return counts >= self.temporal_required

    def diagnostics(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return gravity vectors, validity, alignment errors, and classifications."""

        gravity, gravity_ref, valid, cos_theta = self._get_gravity_info(df)
        raw_violated = np.zeros(valid.shape, dtype=bool)
        raw_violated[valid] = cos_theta[valid] < self.threshold_cos
        violated = raw_violated.copy()

        if self.temporal_filter:
            for bone in self.bones:
                bone_idx = NATNET.index(bone)
                violated[:, bone_idx] = (
                    self._confirm_over_time(violated[:, bone_idx])
                    & valid[:, bone_idx]
                )

        angular_error = np.full(cos_theta.shape, np.nan)
        angular_error[valid] = np.degrees(np.arccos(cos_theta[valid]))
        return {
            "violated": violated,
            "raw_violated": raw_violated,
            "valid": valid,
            "cosine": cos_theta,
            "angular_error": angular_error,
            "gravity": gravity,
            "gravity_reference": gravity_ref,
        }

    def _check(self, df: pd.DataFrame) -> np.ndarray:
        return self.diagnostics(df)["violated"]


class AngularVelocityChecker(BaseChecker):
    """Detect implausibly fast parent-relative NatNet rotations."""

    def __init__(
        self,
        threshold_rad_s: float = 10.0,
        bones: tuple[str, ...] | list[str] | None = None,
        plot: bool = False,
    ):
        self.threshold_rad_s = threshold_rad_s
        self.bones = tuple(NATNET.names) if bones is None else tuple(bones)
        self.plot = plot

    def diagnostics(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return local angular velocities, speeds, and classifications."""

        quats = extract_rotations(df, list(NATNET.names))
        local_quats = global_to_local(quats)
        angular_velocity = np.zeros((len(df), len(NATNET), 3))

        if len(df) > 1:
            dt_seconds = df["timestamp"].diff().dt.total_seconds().to_numpy()
            previous = Rotation.from_quat(local_quats[:-1].reshape(-1, 4))
            current = Rotation.from_quat(local_quats[1:].reshape(-1, 4))
            rotation_vectors = (previous.inv() * current).as_rotvec()
            rotation_vectors = rotation_vectors.reshape(len(df) - 1, len(NATNET), 3)
            angular_velocity[1:] = rotation_vectors / dt_seconds[1:, None, None]

        angular_speed = np.linalg.norm(angular_velocity, axis=2)
        violated = np.zeros((len(df), len(NATNET)), dtype=bool)

        for bone in self.bones:
            bone_idx = NATNET.index(bone)
            violated[:, bone_idx] = angular_speed[:, bone_idx] > self.threshold_rad_s

        return {
            "violated": violated,
            "angular_velocity": angular_velocity,
            "angular_speed": angular_speed,
        }

    def _check(self, df: pd.DataFrame) -> np.ndarray:
        details = self.diagnostics(df)

        if self.plot:
            for bone in self.bones:
                bone_idx = NATNET.index(bone)
                plot_angular_velocity(
                    bone,
                    details["angular_speed"][:, bone_idx],
                    details["violated"][:, bone_idx],
                    self.threshold_rad_s,
                )

        return details["violated"]


class LinearAccelerationChecker(BaseChecker):
    """Detect implausibly large parent-relative NatNet accelerations."""

    def __init__(
        self,
        threshold_m_s2: float | dict[str, float] | None = None,
        bones: tuple[str, ...] | list[str] | None = None,
        plot: bool = False,
    ):
        self.bones = tuple(NATNET.names) if bones is None else tuple(bones)
        if threshold_m_s2 is None:
            self.thresholds = LINEAR_ACCELERATION_LIMITS
        elif isinstance(threshold_m_s2, dict):
            self.thresholds = threshold_m_s2
        else:
            self.thresholds = {bone: threshold_m_s2 for bone in self.bones}
        self.plot = plot

    def diagnostics(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return local linear accelerations, magnitudes, and classifications."""

        quats = extract_rotations(df, list(NATNET.names))
        positions = extract_positions(df, list(NATNET.names))
        _, local_positions = global_to_local(quats, positions)
        velocity = np.zeros_like(local_positions)
        acceleration = np.zeros_like(local_positions)

        if len(df) > 1:
            dt_seconds = df["timestamp"].diff().dt.total_seconds().to_numpy()
            velocity[1:] = np.diff(local_positions, axis=0) / dt_seconds[1:, None, None]

        if len(df) > 2:
            acceleration_dt = (dt_seconds[1:-1] + dt_seconds[2:]) / 2
            acceleration[2:] = np.diff(velocity[1:], axis=0) / acceleration_dt[:, None, None]

        acceleration_magnitude = np.linalg.norm(acceleration, axis=2)
        violated = np.zeros((len(df), len(NATNET)), dtype=bool)

        for bone in self.bones:
            bone_idx = NATNET.index(bone)
            violated[:, bone_idx] = acceleration_magnitude[:, bone_idx] > self.thresholds[bone]

        return {
            "violated": violated,
            "linear_acceleration": acceleration,
            "acceleration_magnitude": acceleration_magnitude,
        }

    def _check(self, df: pd.DataFrame) -> np.ndarray:
        details = self.diagnostics(df)

        if self.plot:
            for bone in self.bones:
                bone_idx = NATNET.index(bone)
                plot_linear_acceleration(
                    bone,
                    details["acceleration_magnitude"][:, bone_idx],
                    details["violated"][:, bone_idx],
                    self.thresholds[bone],
                )

        return details["violated"]
