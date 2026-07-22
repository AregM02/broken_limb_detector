"""Validation checkers used by the break detector."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from src.skeletons.natnet_skeleton import NATNET, extract_rotations
from src.utils.gravity import compute_gravity_vectors, cosine_similarity
from src.utils.transforms import global_to_local
from src.utils.visualization import plot_violations


# Extrinsic rotations around fixed parent frame XYZ axes
ABSOLUTE_LIMITS: dict[str, np.ndarray] = {
    "RHand": np.array([[-45, 33], [-33, 36], [-88, 63]], dtype=float),
    "LHand": np.array([[-50, 43], [-24, 33], [-63, 88]], dtype=float),
    "RFoot": np.array([[-22, 44], [-73, 48.5], [-85, 66]], dtype=float),
    "LFoot": np.array([[-15, 52], [-43, 73], [-23, 43]], dtype=float),
}
END_EFFECTOR_BONES: tuple[str, ...] = tuple(ABSOLUTE_LIMITS)


class BaseChecker(ABC):
    def check(self, df: pd.DataFrame) -> np.ndarray:
        mask = self._check(df)
        expected = (len(df), len(NATNET))
        if mask.shape != expected:
            raise ValueError(f"{self.__class__.__name__} returned mask shape {mask.shape}, expected {expected}")
        if mask.dtype != bool:
            raise TypeError(f"{self.__class__.__name__} must return a boolean mask")
        return mask

    @abstractmethod
    def _check(self, df: pd.DataFrame) -> np.ndarray:
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

    def _check(self, df: pd.DataFrame) -> np.ndarray:
        local_quats, rpy = self._get_quats_and_rpy(df)
        n_frames, n_bones, _ = rpy.shape
        violated = np.zeros((n_frames, n_bones), dtype=bool)

        for key in self.limits:
            bone_idx = NATNET.index(key) if isinstance(key, str) else key
            if bone_idx >= n_bones:
                continue

            bounds = self._effective_bounds(key)
            bone_rpy = rpy[:, bone_idx, :]
            violated[:, bone_idx] = ((bone_rpy < bounds[:, 0]) | (bone_rpy > bounds[:, 1])).any(axis=1)

            if self.plot:
                plot_violations(NATNET[bone_idx], local_quats, violated[:, bone_idx], bounds)

        return violated


class GravityAlignmentChecker(BaseChecker):
    """Compare the gravity-fit NatNet path against constant NatNet-world gravity."""

    def __init__(
        self,
        threshold_cos: float = 0.8,
        calibration_start: int = 0,
        calibration_window: int = 600,
        bones: tuple[str, ...] | list[str] | None = None,
    ):
        self.threshold_cos = threshold_cos
        self.calibration_start = calibration_start
        self.calibration_window = calibration_window
        self.bones = END_EFFECTOR_BONES if bones is None else tuple(bones)
        for bone in self.bones:
            if bone not in NATNET.names:
                raise ValueError(f"unknown NatNet bone {bone!r}")
        self._cached_df: pd.DataFrame | None = None
        self._cached_config: tuple[int, int, tuple[str, ...]] | None = None
        self._cached_valid: np.ndarray | None = None
        self._cached_cosine: np.ndarray | None = None

    def clear_cache(self) -> None:
        self._cached_df = None
        self._cached_config = None
        self._cached_valid = None
        self._cached_cosine = None

    def _get_cosine(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        config = (self.calibration_start, self.calibration_window, self.bones)
        if self._cached_df is df and self._cached_config == config:
            return self._cached_valid, self._cached_cosine

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
        self._cached_valid = valid
        self._cached_cosine = cos_theta
        return valid, cos_theta

    def _check(self, df: pd.DataFrame) -> np.ndarray:
        valid, cos_theta = self._get_cosine(df)
        violated = np.zeros(valid.shape, dtype=bool)
        violated[valid] = cos_theta[valid] < self.threshold_cos
        return violated
