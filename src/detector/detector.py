"""Detector orchestration and reporting."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from src.detector.checkers import (
    ABSOLUTE_LIMITS,
    AbsoluteLimitChecker,
    BaseChecker,
    CheckerResult,
    END_EFFECTOR_BONES,
    GravityAlignmentChecker,
)
from src.skeletons.natnet_skeleton import NATNET


class DetectionResult(NamedTuple):
    """Aggregated detector output."""

    violated: np.ndarray
    violated_axes: np.ndarray
    rpy: np.ndarray
    n_total: int
    n_violations: int
    checker_results: dict[str, CheckerResult]


def default_checkers() -> list[BaseChecker]:
    """Return the standard detector pipeline."""

    return [AbsoluteLimitChecker(), GravityAlignmentChecker()]


def detect_breaks(df: pd.DataFrame, checkers: list[BaseChecker] | None = None) -> DetectionResult:
    """Run detector checks and combine their violation masks."""

    if checkers is None:
        checkers = default_checkers()

    n_frames = len(df)
    n_bones = len(NATNET.names)
    combined_violated = np.zeros((n_frames, n_bones), dtype=bool)
    checker_results: dict[str, CheckerResult] = {}

    rpy_fallback = np.zeros((n_frames, n_bones, 3))
    violated_axes_fallback = np.zeros((n_frames, n_bones, 3), dtype=bool)

    for checker in checkers:
        name = checker.__class__.__name__
        result = checker.check(df)
        checker_results[name] = result
        combined_violated |= result.violated

        if name == "AbsoluteLimitChecker":
            rpy_fallback = result.details.get("rpy", rpy_fallback)
            violated_axes_fallback = result.details.get("violated_axes", violated_axes_fallback)

    return DetectionResult(
        violated=combined_violated,
        violated_axes=violated_axes_fallback,
        rpy=rpy_fallback,
        n_total=n_frames * n_bones,
        n_violations=int(combined_violated.sum()),
        checker_results=checker_results,
    )


def format_report(result: DetectionResult, max_rows: int = 20) -> str:
    """Build a human-readable detector summary."""

    lines: list[str] = [
        f"Total Combined Violations: {result.n_violations} / {result.n_total} "
        f"({100.0 * result.n_violations / result.n_total:.2f}%)"
    ]

    for name, checker_result in result.checker_results.items():
        lines.append(f"  - {name}: flagged {int(checker_result.violated.sum())} broken frames")

    if result.n_violations == 0:
        return "\n".join(lines)

    lines.append("\nFirst flagged violations breakdown:")
    violating_frames, violating_bones = np.where(result.violated)
    for shown, (frame, bone_idx) in enumerate(zip(violating_frames, violating_bones)):
        if shown >= max_rows:
            lines.append(f"  ... and {len(violating_frames) - shown} more")
            break

        triggers = [
            name for name, checker_result in result.checker_results.items()
            if checker_result.violated[frame, bone_idx]
        ]
        rpy = result.rpy[frame, bone_idx]
        lines.append(
            f"  Frame {frame:6d}  {NATNET[bone_idx]:>10s}  "
            f"R({rpy[0]:+6.1f}) P({rpy[1]:+6.1f}) Y({rpy[2]:+6.1f}) deg | "
            f"Triggered by: {', '.join(triggers)}"
        )

    return "\n".join(lines)


__all__ = [
    "ABSOLUTE_LIMITS",
    "AbsoluteLimitChecker",
    "BaseChecker",
    "CheckerResult",
    "END_EFFECTOR_BONES",
    "DetectionResult",
    "GravityAlignmentChecker",
    "default_checkers",
    "detect_breaks",
    "format_report",
]
