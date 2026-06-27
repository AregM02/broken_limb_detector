import matplotlib
# matplotlib.use('gtk3agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from src.skeletons.natnet_skeleton import NATNET, LINKS

_AXIS_LEN = 0.15


_GRAVITY_LEN = 0.2


def plot_skeleton(
    pos: np.ndarray,
    ori: np.ndarray,
    frame: int = 0,
    gravity: np.ndarray | None = None,
    gravity_ref: np.ndarray | None = None,
    gravity_len: float = _GRAVITY_LEN,
    figsize: tuple[int, int] = (14, 10),
) -> None:
    """Draw a 3D skeleton with bone connections, joint axes, labels, and
    optional gravity arrows.

    When *gravity* is provided, each bone gets a blue arrow showing the
    **world-frame** gravity direction.  The caller is responsible for
    providing gravity already expressed in the **world** frame — typically
    by rotating the sensorsuit's local gravity measurement with the
    **sensorsuit's own** orientation quaternion (not the NatNet bone
    orientation, since the IMU and bone frames differ).

    In a correctly tracked skeleton all gravity arrows should point
    consistently.  A bone whose orientation estimate is broken will show
    a gravity arrow that deviates from the others.

    Parameters
    ----------
    pos:
        (n_frames, n_bones, 3) global positions.
    ori:
        (n_frames, n_bones, 4) global orientations in xyzw.
    frame:
        Frame index to plot.
    gravity:
        (n_frames, n_bones, 3) **world-frame** gravity vectors.  If 2-D
        ``(n_bones, 3)`` it is used directly.
    gravity_ref:
        (n_frames, n_bones, 3) or (n_bones, 3) reference gravity vectors
        (e.g. gravity predicted from the sensor suit's own orientation).
        Drawn as cyan arrows at each joint for comparison.
    gravity_len:
        Display length of the gravity arrows (in data coordinates).
    figsize:
        Figure size passed to ``plt.figure()``.
    """
    pos = pos[frame]
    ori = ori[frame]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    p = pos.copy()
    p[:, 0] = -p[:, 0]
    p[:, [1, 2]] = p[:, [2, 1]]

    center = (p.max(axis=0) + p.min(axis=0)) / 2
    extent = max(p.max(axis=0) - p.min(axis=0)) * 0.4
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_zlim(center[2] - extent, center[2] + extent)
    ax.set_axis_off()
    ax.set_title(f"Frame {frame} ({len(pos)} total)")

    bones = list(NATNET.names)
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
        for axis_idx, color in enumerate(['red', 'green', 'blue']):
            v = R[:, axis_idx] * _AXIS_LEN
            v[0] = -v[0]
            v[[1, 2]] = v[[2, 1]]
            ax.plot([p[i, 0], p[i, 0] + v[0]],
                    [p[i, 1], p[i, 1] + v[1]],
                    [p[i, 2], p[i, 2] + v[2]],
                    color=color, alpha=0.5, linewidth=1)
        ax.scatter(*p[i], color='steelblue', s=12)
        ax.text(*p[i], name, size=7, ha='center', va='bottom')

        if grav is not None:
            g = grav[i]
            g_norm = np.linalg.norm(g)
            if g_norm > 1e-8:
                g_dir = g / g_norm * gravity_len
                g_dir[0] = -g_dir[0]
                g_dir[[1, 2]] = g_dir[[2, 1]]
                ax.quiver(*p[i], *g_dir, color='blue', alpha=0.7,
                          linewidth=2, arrow_length_ratio=0.15)

        if grav_ref is not None:
            g = grav_ref[i]
            g_norm = np.linalg.norm(g)
            if g_norm > 1e-8:
                g_dir = g / g_norm * gravity_len
                g_dir[0] = -g_dir[0]
                g_dir[[1, 2]] = g_dir[[2, 1]]
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
    angle = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    if angle < 1e-6:
        return np.empty((0, 3))
    t = np.linspace(0, 1, n)
    return radius * (
        np.sin((1 - t)[:, None] * angle) * a[None, :]
        + np.sin(t[:, None] * angle) * b[None, :]
    ) / np.sin(angle)


def _disp_tf(v: np.ndarray) -> np.ndarray:
    """Apply display transform (negate X, swap Y↔Z) to a vector."""
    out = v.copy()
    out[0] = -out[0]
    out[[1, 2]] = out[[2, 1]]
    return out


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

    def _ax_cols(R):
        cols = R * axlen
        cols[0] = -cols[0]
        cols[[1, 2]] = cols[[2, 1]]
        return cols

    parent_axes = _ax_cols(np.eye(3))
    child_axes = _ax_cols(Rotation.from_quat(q_rel).as_matrix())

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


def plot_child_in_parent(
    child_name: str,
    local_quats: np.ndarray,
) -> None:
    """Plot rotation-axis histogram + axis-specific angle time series for one bone."""
    child_idx = NATNET.index(child_name)
    parent_idx = NATNET.parent_of(child_idx)
    if parent_idx < 0:
        raise ValueError(f"{child_name} is the root — no parent frame.")
    parent_name = NATNET[parent_idx]

    q = local_quats[:, child_idx, :]

    xv, yv, zv = q[:, 0], q[:, 1], q[:, 2]
    norm = np.sqrt(xv*xv + yv*yv + zv*zv) + 1e-12
    ax_x, ax_y, ax_z = xv / norm, yv / norm, zv / norm

    rv = Rotation.from_quat(q).as_rotvec()
    rv_deg = np.rad2deg(rv)

    # Using 6 rows for better vertical control; right side plots span 2 rows each
    fig = plt.figure(figsize=(10, 12), constrained_layout=True)
    gs = fig.add_gridspec(6, 2, width_ratios=[1, 1.8], hspace=0.1)
    
    ax_hist = fig.add_subplot(gs[:, 0])
    # Define subplots with sharex for cleaner look
    ax1 = fig.add_subplot(gs[0:2, 1])
    ax2 = fig.add_subplot(gs[2:4, 1], sharex=ax1)
    ax3 = fig.add_subplot(gs[4:6, 1], sharex=ax1)
    axes = [ax1, ax2, ax3]
    
    colors = ['#d62728', '#2ca02c', '#1f77b4']
    labels = ['x', 'y', 'z']

    # Histogram
    bins = np.linspace(0, 1, 40)
    ax_hist.hist(np.abs(ax_x), bins=bins, alpha=0.5, label='x', color=colors[0])
    ax_hist.hist(np.abs(ax_y), bins=bins, alpha=0.5, label='y', color=colors[1])
    ax_hist.hist(np.abs(ax_z), bins=bins, alpha=0.5, label='z', color=colors[2])
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel("|dot| of rotation axis")
    ax_hist.set_ylabel("frames")
    ax_hist.set_title(f"{child_name} in {parent_name}")
    ax_hist.legend(fontsize=8)

    # Time series
    for ax, rv_i, c, lbl in zip(axes, rv_deg.T, colors, labels):
        ax.plot(rv_i, lw=0.4, color=c, alpha=0.7)
        ax.axhline(0, color='gray', ls='-', lw=0.3)
        ax.set_ylabel(f"angle_{lbl} (deg)")
        
    axes[-1].set_xlabel("frames")
    axes[0].set_title(f"{child_name} rotation per axis")

    plt.show()


def plot_bone_with_violations(
    child_name: str,
    local_quats: np.ndarray | None = None,
    violated: np.ndarray | None = None,
    violated_axes: np.ndarray | None = None,
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    rpy: np.ndarray | None = None,
    gravity_violated: np.ndarray | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> None:
    """Time series of roll/pitch/yaw for one bone with violation markers.

    Parameters
    ----------
    child_name:
        Name of the bone to plot.
    local_quats:
        (n_frames, n_bones, 4) parent-relative quaternions.
        Not needed when *rpy* is provided.
    violated:
        (n_frames,) or (n_frames, n_bones) boolean mask of violations.
        If 2-D, the bone column is selected automatically and red spans
        are drawn on all three axes when *violated_axes* is not given.
    violated_axes:
        (n_frames, n_bones, 3) or (n_frames, 3) per-axis violation mask.
        When given, each subplot only highlights its own axis.
    limits:
        Optional ``((r_lo, r_hi), (p_lo, p_hi), (y_lo, y_hi))`` to draw
        as horizontal dashed lines.
    rpy:
        Pre-computed RPY array (n_frames, n_bones, 3).  Computed from
        *local_quats* when *None*.
    gravity_violated:
        (n_frames,) or (n_frames, n_bones) boolean mask of gravity
        alignment violations.  Drawn as blue vertical spans.
    figsize:
        Figure size passed to ``plt.figure()``.
    """
    child_idx = NATNET.index(child_name)

    if rpy is not None:
        angles = rpy[:, child_idx, :]
    else:
        angles = Rotation.from_quat(local_quats[:, child_idx, None]).as_euler('xyz', degrees=True)[:, 0, :]

    n_frames = len(angles)
    if violated_axes is not None:
        if violated_axes.ndim == 3:
            ax_masks = violated_axes[:, child_idx, :]
        else:
            ax_masks = violated_axes
    elif violated is not None:
        if violated.ndim == 2:
            bone_mask = violated[:, child_idx]
        else:
            bone_mask = violated
        ax_masks = None
    else:
        bone_mask = None
        ax_masks = None

    if gravity_violated is not None:
        if gravity_violated.ndim == 2:
            grav_mask = gravity_violated[:, child_idx]
        else:
            grav_mask = gravity_violated
    else:
        grav_mask = None

    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    axis_labels = ["Roll (X)", "Pitch (Y)", "Yaw (Z)"]
    limit_labels = ["r", "p", "y"]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle(f"{child_name} — RPY time series", fontsize=13)

    for ax, angle_i, color, label, lim_label in zip(
        axes, angles.T, colors, axis_labels, limit_labels
    ):
        ax.plot(angle_i, lw=0.5, color=color, alpha=0.7)
        ax.axhline(0, color="gray", ls="-", lw=0.3)
        ax.set_ylabel(f"{label} (deg)")

        if ax_masks is not None:
            axis_idx = {"r": 0, "p": 1, "y": 2}[lim_label]
            mask = ax_masks[:, axis_idx]
            padded = np.concatenate([[False], mask, [False]])
            transitions = np.diff(padded.astype(int))
            for s, e in zip(np.flatnonzero(transitions > 0), np.flatnonzero(transitions < 0)):
                ax.axvspan(s, e, color="red", alpha=0.12, lw=0)
        elif bone_mask is not None:
            padded = np.concatenate([[False], bone_mask, [False]])
            transitions = np.diff(padded.astype(int))
            for s, e in zip(np.flatnonzero(transitions > 0), np.flatnonzero(transitions < 0)):
                ax.axvspan(s, e, color="red", alpha=0.12, lw=0)

        if grav_mask is not None:
            padded = np.concatenate([[False], grav_mask, [False]])
            transitions = np.diff(padded.astype(int))
            for s, e in zip(np.flatnonzero(transitions > 0), np.flatnonzero(transitions < 0)):
                ax.axvspan(s, e, color="blue", alpha=0.12, lw=0)

        if limits is not None:
            axis_idx = {"r": 0, "p": 1, "y": 2}[lim_label]
            ax.axhline(limits[axis_idx][0], color=color, ls="--", lw=0.8, alpha=0.6)
            ax.axhline(limits[axis_idx][1], color=color, ls="--", lw=0.8, alpha=0.6)

    axes[-1].set_xlabel("Frame")

    plt.tight_layout()
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

