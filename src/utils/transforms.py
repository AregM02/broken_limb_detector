"""NatNet world-frame to parent-relative quaternion transforms."""

import numpy as np
from scipy.spatial.transform import Rotation
from src.skeletons.natnet_skeleton import NATNET


def _effective_parents(use_kinematic_chain: bool = True) -> list[int]:
    if use_kinematic_chain:
        return list(NATNET.parent)
    return [-1 if i == 0 else 0 for i in range(len(NATNET))]


def _as_bone_view(q: np.ndarray, dim: int = 4) -> np.ndarray:
    *leading, last = q.shape
    n = len(NATNET)
    if last == n * dim:
        return q.reshape(*leading, n, dim)
    if last == dim:
        if not leading or leading[-1] != n:
            raise ValueError(f"Expected {n} bones before last dim {dim}, got shape {q.shape}")
        return q
    raise ValueError(f"Expected last dim {n * dim} or {dim}, got {last}")


def global_to_local(
    orientations_xyzw: np.ndarray,
    positions: np.ndarray | None = None,
    use_kinematic_chain: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Convert world-frame orientations (and optionally positions) to
    parent-relative.

    Parameters
    ----------
    orientations_xyzw:
        (..., N*4) or (..., N, 4) global quaternions in xyzw.
    positions:
        (..., N*3) or (..., N, 3) global positions.
        If given, parent-relative positions are returned alongside the quats.
    use_kinematic_chain:
        If True, use the full kinematic chain; otherwise all bones are
        referenced to the root.

    Returns
    -------
    local_quats:
        Same shape as *orientations_xyzw*.
    local_positions:
        Only returned if *positions* is given — same shape as *positions*.
        Each bone's position is rotated into its parent's local frame:
        ``p_local = R_parent⁻¹ @ (p_child - p_parent)``.
    """
    view = _as_bone_view(orientations_xyzw)
    parents = _effective_parents(use_kinematic_chain)
    num_bones = len(NATNET)

    quat_parts: list[np.ndarray] = []
    pos_parts: list[np.ndarray] | None = [] if positions is not None else None

    if positions is not None:
        pos_view = _as_bone_view(positions, dim=3)

    for j in range(num_bones):
        q_j = view[..., j, :]
        p = parents[j]
        if p < 0:
            quat_parts.append(q_j)
            if pos_parts is not None:
                pos_parts.append(pos_view[..., j, :])
        else:
            parent_inv = Rotation.from_quat(view[..., p, :]).inv()
            quat_parts.append((parent_inv * Rotation.from_quat(q_j)).as_quat())
            if pos_parts is not None:
                disp = pos_view[..., j, :] - pos_view[..., p, :]
                pos_parts.append(parent_inv.apply(disp))

    local_quats = np.stack(quat_parts, axis=-2)
    local_quats = local_quats.reshape(orientations_xyzw.shape)

    if pos_parts is not None:
        local_pos = np.stack(pos_parts, axis=-2)
        local_pos = local_pos.reshape(positions.shape)
        return local_quats, local_pos

    return local_quats
