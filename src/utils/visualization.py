import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import geometric_slerp
from scipy.spatial.transform import Rotation
from src.skeletons.natnet_skeleton import LINKS, NATNET, extract_positions, extract_rotations

_AXIS_LEN = 0.15


_GRAVITY_LEN = 0.2


def _disp_tf(v: np.ndarray) -> np.ndarray:
    """Apply display transform (negate X, swap Y/Z) to vectors."""

    out = np.array(v, copy=True)
    out[..., 0] = -out[..., 0]
    out[..., [1, 2]] = out[..., [2, 1]]
    return out


def _disp_axes(matrix: np.ndarray, scale: float) -> np.ndarray:
    return _disp_tf((matrix * scale).T).T


def plot_skeleton(
    df: pd.DataFrame,
    frame: int = 0,
    gravity: np.ndarray | None = None,
    gravity_ref: np.ndarray | None = None,
    figsize: tuple[int, int] = (14, 10),
) -> None:
    """Draw a 3D skeleton with bone connections, joint axes, labels, and
    optional gravity arrows.

    When *gravity* is provided, each bone gets a blue arrow showing a
    world-frame gravity direction.  In the current pipeline this is the
    NatNet path: local IMU gravity is mapped through the fitted SensorSuit ->
    NatNet local transform and then through the NatNet bone orientation.

    In a correctly tracked skeleton all gravity arrows should point
    consistently.  A bone whose orientation estimate is broken will show
    a gravity arrow that deviates from the others.

    Parameters
    ----------
    df:
        DataFrame containing NatNet position and rotation columns.
    frame:
        Frame index to plot.
    gravity:
        (n_frames, n_bones, 3) **world-frame** gravity vectors.  If 2-D
        ``(n_bones, 3)`` it is used directly.
    gravity_ref:
        (n_frames, n_bones, 3) or (n_bones, 3) NatNet-world reference gravity
        vectors. Drawn as cyan arrows for comparison.
    figsize:
        Figure size passed to ``plt.figure()``.
    """
    bones = list(NATNET.names)
    pos = extract_positions(df, bones)[frame]
    ori = extract_rotations(df, bones)[frame]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    p = _disp_tf(pos)

    center = (p.max(axis=0) + p.min(axis=0)) / 2
    extent = max(p.max(axis=0) - p.min(axis=0)) * 0.4
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_zlim(center[2] - extent, center[2] + extent)
    ax.set_axis_off()
    ax.set_title(f"Frame {frame} ({len(df)} total)")

    for child, parent in LINKS:
        ci, pi = bones.index(child), bones.index(parent)
        ax.plot(*zip(p[pi], p[ci]), color='gray', linewidth=2)

    for child, parent in LINKS:
        ci, pi = bones.index(child), bones.index(parent)
        vec = p[ci] - p[pi]
        mid = (p[pi] + p[ci]) / 2
        ax.quiver(*p[pi], *vec, color='orange', alpha=0.8, linewidth=3, arrow_length_ratio=0.12)
        ax.text(*mid, f'{parent}→{child}', size=6, color='darkorange', ha='center', va='bottom')

    # extract per-frame gravity / gravity_ref if provided
    grav = None
    if gravity is not None:
        grav = gravity[frame] if gravity.ndim == 3 else gravity
    grav_ref = None
    if gravity_ref is not None:
        grav_ref = gravity_ref[frame] if gravity_ref.ndim == 3 else gravity_ref

    for i, name in enumerate(bones):
        R = Rotation.from_quat(ori[i]).as_matrix()
        axis_vectors = _disp_axes(R, _AXIS_LEN)
        for axis_idx, color in enumerate(['red', 'green', 'blue']):
            v = axis_vectors[:, axis_idx]
            ax.plot([p[i, 0], p[i, 0] + v[0]],
                    [p[i, 1], p[i, 1] + v[1]],
                    [p[i, 2], p[i, 2] + v[2]],
                    color=color, alpha=0.5, linewidth=1)
        ax.scatter(*p[i], color='steelblue', s=12)
        label_pos = p[i] + axis_vectors.sum(axis=1) * 0.4
        ax.text(*label_pos, name, size=8, ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1))

        if grav is not None:
            g = grav[i]
            g_norm = np.linalg.norm(g)
            if g_norm > 1e-8:
                g_dir = _disp_tf(g / g_norm * _GRAVITY_LEN)
                ax.quiver(*p[i], *g_dir, color='blue', alpha=0.7,
                          linewidth=2, arrow_length_ratio=0.15)

        if grav_ref is not None:
            g = grav_ref[i]
            g_norm = np.linalg.norm(g)
            if g_norm > 1e-8:
                g_dir = _disp_tf(g / g_norm * _GRAVITY_LEN)
                ax.quiver(*p[i], *g_dir, color='cyan', alpha=0.7,
                          linewidth=2, arrow_length_ratio=0.15)

        if grav is not None and grav_ref is not None:
            ga = grav[i]
            gb = grav_ref[i]
            na = np.linalg.norm(ga)
            nb = np.linalg.norm(gb)
            if na > 1e-8 and nb > 1e-8:
                dot = np.clip(np.dot(ga, gb) / (na * nb), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(dot))
                ax.text(*p[i], f"{angle_deg:.1f}°", size=8,
                        color='black', ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white', edgecolor='none', alpha=0.6))

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color='red',   lw=2, label='X'),
        Line2D([0], [0], color='green', lw=2, label='Y'),
        Line2D([0], [0], color='blue',  lw=2, label='Z'),
    ]
    if grav is not None:
        legend_elems.append(Line2D([0], [0], color='blue', lw=2, label='gravity'))
    if gravity_ref is not None:
        legend_elems.append(Line2D([0], [0], color='cyan', lw=3, label='gravity ref'))
    ax.legend(handles=legend_elems, title='local axes', loc='upper right', fontsize=7)

    plt.show()
    plt.close(fig)


def _slerp_arc(a: np.ndarray, b: np.ndarray, radius: float, n: int = 32) -> np.ndarray:
    """3-D spherical arc from unit vector *a* to *b* at *radius*."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1e-12 or b_norm < 1e-12:
        return np.empty((0, 3))
    a = np.asarray(a) / a_norm
    b = np.asarray(b) / b_norm
    angle = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    if angle < 1e-6:
        return np.empty((0, 3))
    return geometric_slerp(a, b, np.linspace(0, 1, n)) * radius


def plot_bone_pair(
    relative_quats: np.ndarray,
    relative_pos: np.ndarray,
    child_name: str,
    frame: int = 0,
    axlen: float = 0.3,
    figsize: tuple[int, int] = (4, 4),
) -> None:
    """Plot child frame relative to the standardised parent frame (identity).

    The parent frame is drawn as the canonical X/Y/Z axes at the origin.
    The child's position and orientation are the **parent-relative** values.

    Parameters
    ----------
    relative_quats:
        (n_frames, n_bones, 4) parent-relative quaternions in xyzw.
    relative_pos:
        (n_frames, n_bones, 3) parent-relative positions.
    child_name:
        Name of the child bone.
    frame:
        Frame index to plot.
    axlen:
        Length of the drawn coordinate axes.
    figsize:
        Figure size passed to ``plt.figure()``.
    """
    q_rel = relative_quats[frame, NATNET.index(child_name)]
    p_rel = relative_pos[frame, NATNET.index(child_name)]

    parent_name = NATNET.parent_name(child_name)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # apply display transform to the relative position (origin = parent)
    pp = np.zeros(3)
    cp = _disp_tf(p_rel)

    # axis bounds around the child
    ext = max(np.abs(cp)) * 1.5 + axlen
    ax.set_xlim(-ext, ext)
    ax.set_ylim(-ext, ext)
    ax.set_zlim(-ext, ext)
    ax.set_axis_off()

    parent_axes = _disp_axes(np.eye(3), axlen)
    child_axes = _disp_axes(Rotation.from_quat(q_rel).as_matrix(), axlen)

    colors = ['red', 'green', 'blue']
    axis_labels = ['X', 'Y', 'Z']

    for i, c in enumerate(colors):
        ax.plot([pp[0], pp[0] + parent_axes[0, i]],
                [pp[1], pp[1] + parent_axes[1, i]],
                [pp[2], pp[2] + parent_axes[2, i]],
                color=c, linewidth=3, solid_capstyle='round')
    ax.scatter(*pp, color='black', s=40)
    ax.text(*pp, parent_name, size=10, ha='center', va='bottom', fontweight='bold')

    ax.quiver(*pp, *cp, color='gray', linewidth=2.5,
              arrow_length_ratio=0.07, alpha=0.7)
    mid = cp / 2
    ax.text(*mid, 'translation', size=8, color='gray', ha='center', va='bottom')

    for i, c in enumerate(colors):
        ax.plot([cp[0], cp[0] + parent_axes[0, i]],
                [cp[1], cp[1] + parent_axes[1, i]],
                [cp[2], cp[2] + parent_axes[2, i]],
                color=c, linewidth=1.2, linestyle='--', alpha=0.5)

    for i, c in enumerate(colors):
        ax.plot([cp[0], cp[0] + child_axes[0, i]],
                [cp[1], cp[1] + child_axes[1, i]],
                [cp[2], cp[2] + child_axes[2, i]],
                color=c, linewidth=3, solid_capstyle='round')
    ax.scatter(*cp, color='black', s=40)
    ax.text(*cp, child_name, size=10, ha='center', va='bottom', fontweight='bold')

    for i, c in enumerate(colors):
        ref = parent_axes[:, i]
        act = child_axes[:, i]
        arc = _slerp_arc(ref, act, axlen * 0.5)
        if len(arc) > 1:
            ax.plot(arc[:, 0] + cp[0], arc[:, 1] + cp[1], arc[:, 2] + cp[2],
                    color=c, linewidth=1.5, alpha=0.7)
            ax.scatter(cp[0] + arc[-1, 0], cp[1] + arc[-1, 1], cp[2] + arc[-1, 2],
                       color=c, s=10, alpha=0.8)

    rv = Rotation.from_quat(q_rel[None]).as_rotvec()[0]
    angles = np.rad2deg(rv)
    ax.text(*(cp + [0, 0, axlen * 1.3]),
            f'X: {angles[0]:.1f}°  Y: {angles[1]:.1f}°  Z: {angles[2]:.1f}°',
            size=8, color='black', ha='center', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color='gray',  lw=2.5, label='translation'),
        Line2D([0], [0], color='red',   lw=3,   label=f'{axis_labels[0]}'),
        Line2D([0], [0], color='green', lw=3,   label=f'{axis_labels[1]}'),
        Line2D([0], [0], color='blue',  lw=3,   label=f'{axis_labels[2]}'),
        Line2D([0], [0], color='gray',  lw=1.2, linestyle='--', label='parent ref'),
    ]
    ax.legend(handles=legend_elems, title=f'{parent_name} → {child_name}',
              loc='upper right', fontsize=7)

    ax.set_title(f'Frame {frame}')
    plt.show()
    plt.close(fig)


def plot_violations(
    bone_name: str,
    local_quats: np.ndarray,
    violated: np.ndarray,
    limits: np.ndarray,
) -> None:
    """Plot parent-relative RPY angles and highlight predicted violations."""

    angles = Rotation.from_quat(local_quats[:, NATNET.index(bone_name)]).as_euler("xyz", degrees=True)
    frames = np.arange(len(local_quats))

    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    for axis, values, label, (lower, upper) in zip(
        axes,
        angles.T,
        ("Roll (X)", "Pitch (Y)", "Yaw (Z)"),
        limits,
    ):
        axis.plot(frames, values, linewidth=0.7)
        axis.scatter(frames[violated], values[violated], color="red", s=6, label="Violation")
        axis.axhline(lower, color="black", linestyle="--", linewidth=0.8)
        axis.axhline(upper, color="black", linestyle="--", linewidth=0.8)
        axis.set_ylabel(f"{label} [deg]")
        axis.grid(alpha=0.2)

    axes[0].legend()
    axes[-1].set_xlabel("Frame")
    fig.suptitle(f"{bone_name} parent-relative RPY")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_angular_velocity(
    bone_name: str,
    angular_speed: np.ndarray,
    violated: np.ndarray,
    threshold_rad_s: float,
) -> None:
    """Plot parent-relative angular speed and highlight detected breaks."""

    frames = np.arange(len(angular_speed))
    fig, axis = plt.subplots(figsize=(12, 3))
    axis.plot(frames, angular_speed, linewidth=0.7)
    axis.scatter(frames[violated], angular_speed[violated], color="red", s=6, label="Violation")
    axis.axhline(threshold_rad_s, color="black", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Frame")
    axis.set_ylabel("Angular speed [rad/s]")
    axis.set_title(f"{bone_name} parent-relative angular speed")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_linear_acceleration(
    bone_name: str,
    acceleration_magnitude: np.ndarray,
    violated: np.ndarray,
    threshold_m_s2: float,
) -> None:
    """Plot parent-relative linear acceleration and highlight detected breaks."""

    frames = np.arange(len(acceleration_magnitude))
    fig, axis = plt.subplots(figsize=(12, 3))
    axis.plot(frames, acceleration_magnitude, linewidth=0.7)
    axis.scatter(
        frames[violated],
        acceleration_magnitude[violated],
        color="red",
        s=6,
        label="Violation",
    )
    axis.axhline(threshold_m_s2, color="black", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Frame")
    axis.set_ylabel("Linear acceleration [m/s^2]")
    axis.set_title(f"{bone_name} parent-relative linear acceleration")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_nn_vs_ss(
    rpy_nn: np.ndarray,
    rpy_ss: np.ndarray,
    *,
    frame: int | None = None,
) -> None:
    """Overlay two RPY traces in three subplots.

    Parameters
    ----------
    rpy_nn:
        (n_frames, 3) first RPY (solid).
    rpy_ss:
        (n_frames, 3) second RPY (dashed).
    frame:
        If given, plot a single 3-D axis triple instead.
    """
    import matplotlib.pyplot as plt

    rpy_nn = np.squeeze(rpy_nn)
    rpy_ss = np.squeeze(rpy_ss)

    if frame is not None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        _AXIS = 0.7
        for i, c in enumerate(["red", "green", "blue"]):
            v = np.eye(3)[:, i] * _AXIS
            ax.quiver(0, 0, 0, *v, color=c, linewidth=3, alpha=0.9,
                      arrow_length_ratio=0.15)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_axis_off()
        ax.set_title(f"frame {frame}")
        plt.show()
        return

    t = np.arange(len(rpy_nn))
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)

    for i, (ax, lbl, c) in enumerate(zip(axes, ["Roll", "Pitch", "Yaw"],
                                          ["red", "green", "blue"])):
        ax.plot(t, rpy_nn[:, i], lw=0.5, color=c, label=f"NN {lbl}")
        ax.plot(t, rpy_ss[:, i], lw=0.8, color=c, ls="--", label=f"SS {lbl}")
        ax.axhline(0, color="gray", lw=0.3)
        ax.set_ylabel(f"{lbl} (°)")
        ax.legend(fontsize=8)
        ax.grid(True)

    axes[-1].set_xlabel("Frame")
    fig.suptitle("NN (solid) vs SS (dashed)", fontsize=13)
    plt.show()
