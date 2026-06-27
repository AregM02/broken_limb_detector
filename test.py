"""Analyze time-mismatch spikes in NN vs SS relative angle."""

import argparse

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

from src.skeletons.natnet_skeleton import (
    NATNET, SENSORSUIT_TO_NATNET, extract_rotations,
)
from scipy.spatial.transform import Rotation


def _quat_to_rpy_continuous(q: np.ndarray) -> np.ndarray:
    """Sign-flip corrected quaternion → euler angles (degrees)."""
    q = q.copy()
    for i in range(1, q.shape[0]):
        dot = np.sum(q[i] * q[i - 1], axis=-1, keepdims=True)
        q[i] = np.where(dot < 0, -q[i], q[i])
    return Rotation.from_quat(q).as_euler('xyz', degrees=True)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
    return lfilter(b, a, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_path")
    parser.add_argument("--bone", default="hand_right")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Lowpass cutoff Hz")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_path)
    n_frames = len(df)

    nn_quats = extract_rotations(df, list(NATNET.names))
    ss_quats = extract_rotations(df, [args.bone], stream="sensorsuit")[:, 0]

    nn_name = SENSORSUIT_TO_NATNET.get(args.bone)
    nn_i = NATNET.index(nn_name)

    q_nn = nn_quats[:, nn_i]
    q_ss = ss_quats
    q_rel = (Rotation.from_quat(q_nn) * Rotation.from_quat(q_ss).inv()).as_quat()

    rpy_rel = _quat_to_rpy_continuous(q_rel)          # (F, 3)
    rpy_nn = _quat_to_rpy_continuous(q_nn)            # (F, 3)
    rpy_ss = _quat_to_rpy_continuous(q_ss)            # (F, 3)

    # Angular speed (deg/frame)
    dot_q = np.clip(np.abs(np.sum(q_ss[1:] * q_ss[:-1], axis=1)), 0, 1)
    ang_speed = np.zeros(n_frames)
    ang_speed[1:] = np.rad2deg(2 * np.arccos(dot_q))

    # Time axis (assume 100 Hz)
    fs = 100.0
    t = np.arange(n_frames) / fs

    # Lowpass filter relative RPY
    rpy_rel_filt = np.zeros_like(rpy_rel)
    for i in range(3):
        rpy_rel_filt[:, i] = butter_lowpass_filter(rpy_rel[:, i], args.cutoff, fs)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), constrained_layout=True)

    # 1) Raw relative RPY
    ax = axes[0]
    for i, (lbl, c) in enumerate(zip(["Roll", "Pitch", "Yaw"], ["red", "green", "blue"])):
        ax.plot(t, rpy_rel[:, i], lw=0.4, color=c, alpha=0.5, label=f"raw {lbl}")
        ax.plot(t, rpy_rel_filt[:, i], lw=0.8, color=c, label=f"filtered {lbl}")
    ax.set_ylabel("Relative angle (°)")
    ax.set_title(f"{args.bone} → {nn_name}: raw vs filtered (cutoff={args.cutoff} Hz)")
    ax.legend(fontsize=7)
    ax.grid(True)

    # 2) d_roll of NN and SS
    ax = axes[1]
    d_rpy_nn = np.diff(rpy_nn, axis=0)
    d_rpy_ss = np.diff(rpy_ss, axis=0)
    ax.plot(t[1:], d_rpy_nn[:, 0], lw=0.5, color="red", label="NN d_roll")
    ax.plot(t[1:], d_rpy_ss[:, 0], lw=0.5, color="blue", label="SS d_roll")
    ax.set_ylabel("d_roll (°/frame)")
    ax.set_title("d_roll: NN vs SS (spikes = time mismatch)")
    ax.legend(fontsize=8)
    ax.grid(True)

    # 3) NN d_roll vs SS d_roll scatter (should be line if synced)
    ax = axes[2]
    ax.scatter(d_rpy_nn[:, 0], d_rpy_ss[:, 0], s=1, alpha=0.3, c="purple")
    lim = max(np.abs(ax.get_xlim()).max(), np.abs(ax.get_ylim()).max())
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.5)
    ax.set_xlabel("NN d_roll")
    ax.set_ylabel("SS d_roll")
    ax.set_title("d_roll correlation (diagonal = perfect sync)")
    ax.grid(True)

    # 4) Relative roll vs angular speed
    ax = axes[3]
    ax.plot(t, np.abs(rpy_rel[:, 0]), lw=0.5, color="purple", alpha=0.6, label="|relative roll|")
    ax_twin = ax.twinx()
    ax_twin.plot(t, ang_speed, lw=0.5, color="orange", alpha=0.6, label="SS angular speed")
    ax_twin.set_ylabel("Angular speed (°/frame)", color="orange")
    ax_twin.tick_params(axis="y", colors="orange")
    ax.set_ylabel("|relative roll| (°)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Spikes in relative roll coincide with motion")
    ax.legend(fontsize=7, loc="upper left")
    ax_twin.legend(fontsize=7, loc="upper right")
    ax.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
