import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import butter, lfilter

from src.skeletons.natnet_skeleton import (
    SENSORSUIT_TO_NATNET,
    extract_rotations,
    extract_gravity,
    extract_positions
)

def butter_lowpass_filter(data, cutoff=1, fs=100.0, order=6):
    b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
    return lfilter(b, a, data, axis=0)


def main(df, bone_ss):
    bone_nn = SENSORSUIT_TO_NATNET.get(bone_ss)
    if bone_nn is None:
        print(f"No NatNet mapping for '{bone_ss}'")
        return

    q_nn = extract_rotations(df, ['LHand'], stream='natnet').squeeze() #LHand 
    q_ss = extract_rotations(df, ['hand_left'], suffix='rotation', stream='sensorsuit').squeeze() #hand_left
    g_ss = extract_gravity(df, ['hand_left'], stream="sensorsuit").squeeze()

    nn_R_ss = np.array([[0,  -1,  0],
                        [0,  0,  1],
                        [-1,  0,  0]])

    w_R_nn = Rotation.from_quat(q_nn).as_matrix()
    
    g_nn = np.einsum('ij,nj->ni', nn_R_ss, g_ss)
    g_w = np.einsum('nij,nj->ni', w_R_nn, g_nn)

    w_R_ss = Rotation.from_quat(q_ss).as_matrix()
    g_true = np.einsum('nij,nj->ni', w_R_ss, g_ss)
    g_true = np.einsum('ij,nj->ni', nn_R_ss, g_true)

    g_w_norm = g_w / (np.linalg.norm(g_w, axis=1, keepdims=True) + 1e-8)
    g_true_norm = g_true / (np.linalg.norm(g_true, axis=1, keepdims=True) + 1e-8)
    cos_theta = np.clip(np.sum(g_w_norm * g_true_norm, axis=1), -1.0, 1.0)

    # plt.plot(g_w[:, 1], label='g_nn_y')
    # plt.plot(g_true[:, 1], label='g_true_z')
    plt.plot(cos_theta, label = 'cosine similarity - nn*R_ss*g_ss vs g_')
    plt.legend()
    plt.show()
    quit()

    # # === BEGIN optional: skeleton visualisation ============================
    from src.skeletons.natnet_skeleton import NATNET
    from src.utils.visualization import plot_skeleton
    q_all = extract_rotations(df, list(NATNET.names), stream='natnet')
    r_all = extract_positions(df, list(NATNET.names), stream='natnet')
    natnet_to_ss = {v: k for k, v in SENSORSUIT_TO_NATNET.items()}
    nn_R_ss = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    n_frames = len(df)
    n_bones = len(NATNET.names)
    g_w_all = np.zeros((n_frames, n_bones, 3))
    g_true_all = np.zeros((n_frames, n_bones, 3))
    for bone_idx, bone_nn in enumerate(NATNET.names):
        bone_ss = natnet_to_ss.get(bone_nn)
        if bone_ss is None:
            continue
        q_ss_b = extract_rotations(df, [bone_ss], suffix='rotation', stream='sensorsuit')
        if q_ss_b.shape[1] == 0:
            continue
        q_ss_b = q_ss_b[:, 0]
        g_ss_b = extract_gravity(df, [bone_ss], stream='sensorsuit')[:, 0]
        q_nn_b = extract_rotations(df, [bone_nn], stream='natnet')[:, 0]
        w_R_nn = Rotation.from_quat(q_nn_b).as_matrix()
        g_nn_b = np.einsum('ij,nj->ni', nn_R_ss, g_ss_b)
        g_w_all[:, bone_idx] = np.einsum('nij,nj->ni', w_R_nn, g_nn_b)
        w_R_ss = Rotation.from_quat(q_ss_b).as_matrix()
        g_t = np.einsum('nij,nj->ni', w_R_ss, g_ss_b)
        g_true_all[:, bone_idx] = np.einsum('ij,nj->ni', nn_R_ss, g_t)
    plot_skeleton(r_all, q_all, frame=1000, gravity=g_w_all, gravity_ref=g_true_all)
    # # === END   optional: skeleton visualisation ============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_path")
    parser.add_argument("--bone", default="hand_right")
    args = parser.parse_args()

    main(pd.read_parquet(args.parquet_path), args.bone)