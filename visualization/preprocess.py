import numpy as np
import pandas as pd


def process_sequence(seq):
    # 1. Linear Interpolation for dropped tracking
    lh_missing = np.sum(np.abs(seq[:, 132:195]), axis=1) == 0
    rh_missing = np.sum(np.abs(seq[:, 195:258]), axis=1) == 0
    seq[lh_missing, 132:195] = np.nan
    seq[rh_missing, 195:258] = np.nan
    seq = pd.DataFrame(seq).interpolate(method='linear', limit_area='inside').fillna(0.0).to_numpy()

    # 2. Spatial Normalization (Nose center & shoulder distance scaling)
    all_x = np.concatenate([np.arange(0, 132, 4), np.arange(132, 195, 3), np.arange(195, 258, 3)])
    all_y = np.concatenate([np.arange(1, 132, 4), np.arange(133, 195, 3), np.arange(196, 258, 3)])
    all_z = np.concatenate([np.arange(2, 132, 4), np.arange(134, 195, 3), np.arange(197, 258, 3)])

    for i in range(len(seq)):
        nose_x, nose_y, nose_z = seq[i, 0], seq[i, 1], seq[i, 2]
        shoulder_dist = np.sqrt((seq[i, 44] - seq[i, 48]) ** 2 + (seq[i, 45] - seq[i, 49]) ** 2)
        if shoulder_dist == 0:
            shoulder_dist = 1.0

        seq[i, all_x] = (seq[i, all_x] - nose_x) / shoulder_dist
        seq[i, all_y] = (seq[i, all_y] - nose_y) / shoulder_dist
        seq[i, all_z] = (seq[i, all_z] - nose_z) / shoulder_dist

    # 3. Velocity Calculation
    velocity = np.zeros_like(seq)
    velocity[1:] = seq[1:] - seq[:-1]

    return np.concatenate([seq, velocity], axis=-1)


def pad_or_truncate(processed_seq, max_frames=40):
    num_frames, feat_dim = processed_seq.shape
    padded_seq = np.zeros((max_frames, feat_dim), dtype=np.float32)

    if num_frames > max_frames:
        start = (num_frames - max_frames) // 2
        padded_seq[:] = processed_seq[start:start + max_frames, :]
    else:
        start = (max_frames - num_frames) // 2
        padded_seq[start:start + num_frames, :] = processed_seq

    return padded_seq