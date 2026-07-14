"""Skeleton schema definitions and DataFrame column extractors.

Provides the ``Skeleton`` hierarchy class, plus concrete definitions for
both the NatNet (19-bone) and sensorsuit bone sets, bone-name mappings
between them, and helpers to extract rotation / position / gravity
arrays from parquet DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# Skeleton schema - immutable hierarchy definition
@dataclass(frozen=True)
class Skeleton:
    """Immutable skeleton hierarchy definition.

    Usage
    -----
    >>> NATNET.names           # tuple of bone names
    >>> NATNET.parent          # tuple of parent indices (-1 = root)
    >>> NATNET.children        # tuple of child-index tuples
    >>> NATNET[0]              # name of bone 0 → "Hip"
    >>> NATNET.index("Hip")    # index of "Hip" → 0
    >>> NATNET.parent_of(4)    # parent index of bone 4 → 2
    >>> NATNET.children_of(0)  # child indices of Hip → (1, 12, 13)
    >>> len(NATNET)            # → 19
    """

    names: tuple[str, ...]
    parent: tuple[int, ...]

    def __post_init__(self) -> None:
        n = len(self.names)
        if n != len(self.parent):
            raise ValueError(f"names ({n}) and parent ({len(self.parent)}) must have same length")
        if len(set(self.names)) != n:
            raise ValueError(f"bone names must be unique, got {self.names}")
        for i, (name, p) in enumerate(zip(self.names, self.parent)):
            if not (0 <= p < n) and p != -1:
                raise ValueError(
                    f"bone {i} ({name}) has invalid parent index {p}; "
                    f"expected -1 (root) or 0..{n - 1}"
                )
        # compute children lookup once
        kids: tuple[list[int], ...] = tuple([] for _ in range(n))
        for child, p in enumerate(self.parent):
            if p >= 0:
                kids[p].append(child)
        object.__setattr__(self, "_children", tuple(tuple(k) for k in kids))
        object.__setattr__(self, "_name_to_idx", {n: i for i, n in enumerate(self.names)})
        object.__setattr__(self, "_num", n)

    def __len__(self) -> int:
        return self._num

    def __getitem__(self, idx: int) -> str:
        return self.names[idx]

    def index(self, name: str) -> int:
        return self._name_to_idx[name]

    def parent_of(self, idx: int) -> int:
        return self.parent[idx]

    def parent_name(self, name: str) -> str | None:
        """Parent bone name, or None if *name* is the root."""
        p = self.parent_of(self.index(name))
        return None if p < 0 else self.names[p]

    def children_of(self, idx: int) -> tuple[int, ...]:
        return self._children[idx]

    def children_names(self, name: str) -> tuple[str, ...]:
        """Child bone names of *name*."""
        return tuple(self.names[i] for i in self._children[self.index(name)])

    @property
    def children(self) -> tuple[tuple[int, ...], ...]:
        return self._children

    @classmethod
    def from_parent_pairs(cls, pairs: list[tuple[str, str | None]]) -> Skeleton:
        """Build from an ordered list of ``(name, parent_name)`` pairs.

        The first entry in *pairs* whose parent is ``None`` is treated as the
        root bone.
        """
        names = tuple(p[0] for p in pairs)
        name_to_idx = {n: i for i, n in enumerate(names)}
        parent = tuple(-1 if p[1] is None else name_to_idx[p[1]] for p in pairs)
        return cls(names=names, parent=parent)


# NatNet (19-bone) skeleton  —  single source of truth
NATNET_DEF: list[tuple[str, str | None]] = [
    ("Hip", None),
    ("Ab", "Hip"),
    ("Chest", "Ab"),
    ("Neck", "Chest"),
    ("LShoulder", "Chest"),
    ("RShoulder", "Chest"),
    ("LUArm", "LShoulder"),
    ("RUArm", "RShoulder"),
    ("LFArm", "LUArm"),
    ("RFArm", "RUArm"),
    ("LHand", "LFArm"),
    ("RHand", "RFArm"),
    ("LThigh", "Hip"),
    ("RThigh", "Hip"),
    ("LShin", "LThigh"),
    ("RShin", "RThigh"),
    ("LFoot", "LShin"),
    ("RFoot", "RShin"),
    ("Head", "Neck"),
]

NATNET = Skeleton.from_parent_pairs(NATNET_DEF)

LINKS: list[tuple[str, str]] = [
    (child, NATNET.names[parent])
    for child, parent in zip(NATNET.names, NATNET.parent)
    if parent >= 0
]

# Sensorsuit skeleton  (body / v3_fullbody bones)
SENSORSUIT_DEF: list[tuple[str, str | None]] = [
    ("lower_back", None),
    ("chest", "lower_back"),
    ("upper_back", "chest"),
    ("head", "upper_back"),
    ("shoulder_left", "chest"),
    ("upper_arm_left", "shoulder_left"),
    ("forearm_left", "upper_arm_left"),
    ("hand_left", "forearm_left"),
    ("shoulder_right", "chest"),
    ("upper_arm_right", "shoulder_right"),
    ("forearm_right", "upper_arm_right"),
    ("hand_right", "forearm_right"),
    ("chest_left", "chest"),
    ("chest_right", "chest"),
    ("pelvis_left", "lower_back"),
    ("thigh_left", "pelvis_left"),
    ("shank_left", "thigh_left"),
    ("foot_left", "shank_left"),
    ("pelvis_right", "lower_back"),
    ("thigh_right", "pelvis_right"),
    ("shank_right", "thigh_right"),
    ("foot_right", "shank_right"),
]

SENSORSUIT = Skeleton.from_parent_pairs(SENSORSUIT_DEF)

SENSORSUIT_TO_NATNET: dict[str, str] = {
    "chest": "Chest",
    "lower_back": "Hip",
    "upper_back": "Chest",
    "head": "Head",
    "shoulder_left": "LShoulder",
    "upper_arm_left": "LUArm",
    "forearm_left": "LFArm",
    "hand_left": "LHand",
    "shoulder_right": "RShoulder",
    "upper_arm_right": "RUArm",
    "forearm_right": "RFArm",
    "hand_right": "RHand",
    "chest_left": "Chest",
    "chest_right": "Chest",
    "pelvis_left": "Hip",
    "thigh_left": "LThigh",
    "shank_left": "LShin",
    "foot_left": "LFoot",
    "pelvis_right": "Hip",
    "thigh_right": "RThigh",
    "shank_right": "RShin",
    "foot_right": "RFoot",
}

# Column extractors  (stream-based, default to ``natnet_*``)
def extract(
    df: "pd.DataFrame",
    bones: list[str],
    field: str,
    *,
    stream: str = "natnet",
    axes: str = "xyz",
    missing: str = "drop",
) -> np.ndarray:
    """Read ``{stream}_{bone}_{field}_{axis}`` columns into an array."""

    if missing not in {"drop", "raise"}:
        raise ValueError("missing must be 'drop' or 'raise'")

    selected = [
        bone
        for bone in bones
        if all(f"{stream}_{bone}_{field}_{axis}" in df.columns for axis in axes)
    ]
    if missing == "raise" and len(selected) != len(bones):
        missing_cols = [
            f"{stream}_{bone}_{field}_{axis}"
            for bone in bones
            for axis in axes
            if f"{stream}_{bone}_{field}_{axis}" not in df.columns
        ]
        raise KeyError(f"missing columns: {missing_cols}")

    if not selected:
        return np.empty((len(df), 0, len(axes)))

    cols = [f"{stream}_{bone}_{field}_{axis}" for bone in selected for axis in axes]
    return df[cols].to_numpy().reshape(-1, len(selected), len(axes))


def extract_rotations(
    df: "pd.DataFrame",
    bones: list[str],
    stream: str = "natnet",
    suffix: str | None = None,
) -> np.ndarray:
    """Read quaternion columns in xyzw order."""

    if suffix is None:
        suffix = "orientation" if stream == "sensorsuit" else "rotation"
    return extract(df, bones, suffix, stream=stream, axes="xyzw")


def extract_positions(df: "pd.DataFrame", bones: list[str], stream: str = "natnet") -> np.ndarray:
    """Read position columns in xyz order."""

    return extract(df, bones, "position", stream=stream, axes="xyz", missing="raise")


def extract_gravity(df: "pd.DataFrame", bones: list[str], stream: str = "sensorsuit") -> np.ndarray:
    """Read gravity-vector columns in xyz order."""

    return extract(df, bones, "gravity", stream=stream, axes="xyz")
