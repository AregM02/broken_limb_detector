"""Detector orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.detector.checkers import AbsoluteLimitChecker, GravityAlignmentChecker
from src.skeletons.natnet_skeleton import NATNET


def detect(df: pd.DataFrame, checkers: list[object] | None = None) -> pd.DataFrame:
    """Return per-frame broken-bone predictions."""

    if checkers is None:
        checkers = [AbsoluteLimitChecker(), GravityAlignmentChecker()]

    bones = list(NATNET.names)
    broken_bones = np.zeros((len(df), len(bones)), dtype=bool)
    frame_checkers: list[set[str]] = [set() for _ in range(len(df))]

    for checker in checkers:
        violated = checker.check(df)
        checker_name = checker.__class__.__name__
        broken_bones |= violated  # OR combination
        for frame in np.flatnonzero(violated.any(axis=1)):
            frame_checkers[int(frame)].add(checker_name)

    result = pd.DataFrame(broken_bones, columns=[f"Broken{bone}" for bone in bones])
    result["broken"] = broken_bones.any(axis=1)
    result["checkers"] = [sorted(names) for names in frame_checkers]
    return result
