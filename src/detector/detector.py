"""Orchestrator: load a parquet, execute modular validation checks (RPY limits, 
gravity alignment), and flag frames where violations occur."""

from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import argparse
from abc import ABC, abstractmethod
from typing import NamedTuple, Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from src.skeletons.natnet_skeleton import (
    NATNET,
    SENSORSUIT_TO_NATNET,
    extract_rotations,
    extract_positions,
    extract_gravity
)
from src.utils.transforms import global_to_local
from src.utils.visualization import plot_bone_with_violations


# Absolute RPY limits in degrees (centered around zero T-pose).
ABSOLUTE_LIMITS: dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {
    "RHand": ((-45, 45), (-45, 40), (-105, 80)),
    "LHand": ((-45, 45), (-40, 45), (-80, 105)),
    "RFoot": ((-25, 50), (-90, 40), (-60, 40)),
    "LFoot": ((-25, 50), (-40, 90), (-40, 60)),
}

# Reverse map for checking NatNet bones against their SensorSuit counterparts
NATNET_TO_SENSORSUIT = {v: k for k, v in SENSORSUIT_TO_NATNET.items()}


class CheckerResult(NamedTuple):
    """Result payload from an individual checker module."""
    violated: np.ndarray       # (n_frames, n_bones) bool
    details: dict[str, Any]    # Extra diagnostic info specific to the check


class DetectionResult(NamedTuple):
    """Aggregated detection summary containing results from all checkers."""
    violated: np.ndarray          # (n_frames, n_bones) bool — Combined mask (any check)
    violated_axes: np.ndarray     # (n_frames, n_bones, 3) bool — For RPY backwards compatibility
    rpy: np.ndarray               # (n_frames, n_bones, 3) deg — For plotting/reporting compatibility
    n_total: int
    n_violations: int
    checker_results: dict[str, CheckerResult]


# ===========================================================================
# Modular Checkers Base & Implementation
# ===========================================================================

class BaseChecker(ABC):
    """Abstract Base Class for kinematic and inertial breakdown checks."""
    
    @abstractmethod
    def check(self, df: pd.DataFrame) -> CheckerResult:
        """Run verification checks on the provided data tracking frame."""
        pass


class AbsoluteLimitChecker(BaseChecker):
    """Validates joint angles against explicit absolute RPY bounding-box constraints."""

    def __init__(
        self, 
        limits: dict[str | int, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] | None = None,
        calibrate: bool = True,
        tpose_window: int = 600
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

        rpy = Rotation.from_quat(local_quats).as_euler('xyz', degrees=True)
        n_frames, n_bones, _ = rpy.shape
        
        violated = np.zeros((n_frames, n_bones), dtype=bool)
        violated_axes = np.zeros((n_frames, n_bones, 3), dtype=bool)

        for key, axes in self.limits.items():
            bone_idx = NATNET.index(key) if isinstance(key, str) else key
            if bone_idx >= n_bones:
                continue
                
            (r_lo, r_hi), (p_lo, p_hi), (y_lo, y_hi) = axes
            bp = rpy[:, bone_idx, :]
            
            ax_viol = np.stack([
                (bp[:, 0] < r_lo) | (bp[:, 0] > r_hi),
                (bp[:, 1] < p_lo) | (bp[:, 1] > p_hi),
                (bp[:, 2] < y_lo) | (bp[:, 2] > y_hi),
            ], axis=1)
            
            violated_axes[:, bone_idx] = ax_viol
            violated[:, bone_idx] = ax_viol.any(axis=1)

        return CheckerResult(
            violated=violated,
            details={"violated_axes": violated_axes, "rpy": rpy, "limits": self.limits}
        )


class GravityAlignmentChecker(BaseChecker):
    """
    Compares the orientation of the tracking bone frame against IMU gravity.
    Defaults to Z-axis (2) for gravity alignment.
    """
    def __init__(self, threshold_cos: float = 0.985):
        self.threshold_cos = threshold_cos
        # Standard Mocap -> Sensor alignment matrix
        # Right-side transform
        self.nn_R_ss_right = np.array([[0,  1,  0],
                                       [0,  0,  1],
                                       [1,  0,  0]])
        # Left-side transform
        self.nn_R_ss_left = np.array([[0,  -1,  0],
                                      [0,  0,  1],
                                      [-1,  0,  0]])

    def check(self, df: pd.DataFrame) -> CheckerResult:
        n_frames = len(df)
        n_bones = len(NATNET.names)
        violated = np.zeros((n_frames, n_bones), dtype=bool)
        angular_errors = np.zeros((n_frames, n_bones))

        for bone_idx, bone_nn in enumerate(NATNET.names):
            bone_ss = NATNET_TO_SENSORSUIT.get(bone_nn)
            if bone_ss is None:
                continue 

            # Select the correct alignment matrix based on side
            nn_R_ss = self.nn_R_ss_left if bone_nn.startswith('L') else self.nn_R_ss_right

            try:
                q_nn = extract_rotations(df, [bone_nn], stream='natnet').squeeze()
                q_ss = extract_rotations(df, [bone_ss], suffix='rotation', stream='sensorsuit').squeeze()
                g_ss = extract_gravity(df, [bone_ss], stream="sensorsuit").squeeze()
                
                # Align SS gravity to World frame via NatNet/Mocap rotation
                w_R_nn = Rotation.from_quat(q_nn).as_matrix()
                g_nn = np.einsum('ij,nj->ni', nn_R_ss, g_ss)
                g_w = np.einsum('nij,nj->ni', w_R_nn, g_nn)

                # Reference gravity (The "Truth" vector in world frame)
                # We expect the sensor gravity to transform into the World Z-axis (or configured axis)
                w_R_ss = Rotation.from_quat(q_ss).as_matrix()
                g_true = np.einsum('nij,nj->ni', w_R_ss, g_ss)
                g_true = np.einsum('ij,nj->ni', nn_R_ss, g_true)

                # Robust Angle Calculation
                # Normalize vectors to ensure we are only comparing direction
                g_w_norm = g_w / (np.linalg.norm(g_w, axis=1, keepdims=True) + 1e-8)
                g_true_norm = g_true / (np.linalg.norm(g_true, axis=1, keepdims=True) + 1e-8)
                
                cos_theta = np.clip(np.sum(g_w_norm * g_true_norm, axis=1), -1.0, 1.0)

                angular_errors[:, bone_idx] = np.degrees(np.arccos(cos_theta))
                violated[:, bone_idx] = cos_theta < self.threshold_cos

            except Exception:
                continue

        return CheckerResult(violated=violated, details={"angular_errors": angular_errors})


# ===========================================================================
# Core Orchestration Flow
# ===========================================================================

def detect_breaks(df: pd.DataFrame, checkers: list[BaseChecker] | None = None) -> DetectionResult:
    """Runs dataframes sequentially across selected analytical verification blocks.

    Parameters
    ----------
    df: Data source framework.
    checkers: Validation strategy modules to run. Defaults to running 
              AbsoluteLimitChecker if none are declared.
    """
    if not checkers:
        checkers = [AbsoluteLimitChecker()]

    n_frames = len(df)
    n_bones = len(NATNET.names)
    
    # Combined master flag matrix
    combined_violated = np.zeros((n_frames, n_bones), dtype=bool)
    checker_results: dict[str, CheckerResult] = {}

    # Initialize fallbacks for backwards compatibility with visualization functions
    rpy_fallback = np.zeros((n_frames, n_bones, 3))
    violated_axes_fallback = np.zeros((n_frames, n_bones, 3), dtype=bool)
    limits_used = ABSOLUTE_LIMITS

    for checker in checkers:
        name = checker.__class__.__name__
        res = checker.check(df)
        checker_results[name] = res
        combined_violated |= res.violated

        # Harvest properties to fulfill historical dependency hooks safely
        if name == "AbsoluteLimitChecker":
            rpy_fallback = res.details.get("rpy", rpy_fallback)
            violated_axes_fallback = res.details.get("violated_axes", violated_axes_fallback)
            limits_used = res.details.get("limits", limits_used)

    return DetectionResult(
        violated=combined_violated,
        violated_axes=violated_axes_fallback,
        rpy=rpy_fallback,
        n_total=n_frames * n_bones,
        n_violations=int(combined_violated.sum()),
        checker_results=checker_results
    )


def format_report(result: DetectionResult, max_rows: int = 20) -> str:
    """Human-readable summary of modular violations."""
    lines: list[str] = []
    lines.append(
        f"Total Combined Violations: {result.n_violations} / {result.n_total} "
        f"({100.0 * result.n_violations / result.n_total:.2f}%)"
    )
    
    for name, res in result.checker_results.items():
        lines.append(f"  └─ {name}: flagged {int(res.violated.sum())} broken frames")

    if result.n_violations == 0:
        return "\n".join(lines)

    lines.append("\nFirst flagged violations breakdown:")
    violating_frames, violating_bones = np.where(result.violated)
    shown = 0
    for f, b in zip(violating_frames, violating_bones):
        if shown >= max_rows:
            lines.append(f"  ... and {len(violating_frames) - shown} more")
            break
        
        # Check who triggered it
        triggers = [n for n, r in result.checker_results.items() if r.violated[f, b]]
        r = result.rpy[f, b]
        lines.append(
            f"  Frame {f:6d}  {NATNET[b]:>10s}  "
            f"R({r[0]:+6.1f}) P({r[1]:+6.1f}) Y({r[2]:+6.1f})° | Triggered by: {', '.join(triggers)}"
        )
        shown += 1
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect joint breaks in NatNet data using modular strategies.")
    parser.add_argument("parquet_path", help="Path to a NatNet parquet file.")
    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Subtract T-pose quaternion offset before checking RPY limits.",
    )
    parser.add_argument(
        "--tpose_window",
        type=int,
        default=600,
        help="T-pose calibration window in frames.",
    )
    parser.add_argument(
        "--gravity_threshold",
        type=float,
        default=0.8,
        help="Minimum cosine similarity for gravity alignment.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=20,
        help="Max violation rows to print.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show violation plots for each bone.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_path)

    # Initialize active pipelines
    active_checkers: list[BaseChecker] = [
        AbsoluteLimitChecker(calibrate=args.calibrate, tpose_window=args.tpose_window),
        GravityAlignmentChecker(threshold_cos=args.gravity_threshold)
    ]

    result = detect_breaks(df, checkers=active_checkers)
    print(format_report(result, max_rows=args.max_rows))

    grav_violated = None
    if "GravityAlignmentChecker" in result.checker_results:
        grav_violated = result.checker_results["GravityAlignmentChecker"].violated

    if args.plot and "AbsoluteLimitChecker" in result.checker_results:
        limits_config = result.checker_results["AbsoluteLimitChecker"].details["limits"]
        for bone_name in limits_config:
            plot_bone_with_violations(
                bone_name,
                violated=result.violated,
                violated_axes=result.violated_axes,
                limits=limits_config[bone_name],
                rpy=result.rpy,
                gravity_violated=grav_violated,
            )