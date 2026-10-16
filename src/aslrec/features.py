"""Feature engineering on MediaPipe hand landmarks (pure NumPy, no TensorFlow).

Landmark array shape is (..., 21, 3) with (x, y, z) per joint. Index layout:
0 wrist; 1-4 thumb; 5-8 index; 9-12 middle; 13-16 ring; 17-20 pinky.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

EPS = 1e-6
NUM_LANDMARKS = 21
FINGERS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}
TIPS = [4, 8, 12, 16, 20]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
# Parent joint used to build "bone vector" node features for the graph model.
PARENT = [0, 0, 1, 2, 3, 0, 5, 6, 7, 5, 9, 10, 11, 9, 13, 14, 15, 13, 17, 18, 19]


def normalize_landmarks(lm: np.ndarray) -> np.ndarray:
    """Translate the wrist to the origin and scale by the wrist -> middle-MCP distance.

    This removes hand position and hand size (distance to the camera) but keeps
    the orientation of the hand, which matters for ASL letters.
    """
    lm = np.asarray(lm, dtype=np.float32)
    centered = lm - lm[..., 0:1, :]
    scale = np.linalg.norm(centered[..., 9, :], axis=-1)[..., None, None]
    return centered / np.maximum(scale, EPS)


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    return (a * b).sum(-1) / np.maximum(na * nb, EPS)


def joint_angles(n: np.ndarray) -> np.ndarray:
    """Cosine of the angle between consecutive bones of every finger (5 x 3 = 15)."""
    out = []
    for chain in FINGERS.values():
        pts = [0] + chain
        bones = [n[..., pts[i + 1], :] - n[..., pts[i], :] for i in range(4)]
        for i in range(3):
            out.append(_cos(bones[i], bones[i + 1]))
    return np.stack(out, axis=-1)


def spread_angles(n: np.ndarray) -> np.ndarray:
    """Cosine of the angle between neighbouring fingers at the palm (4)."""
    dirs = [n[..., i, :] - n[..., 0, :] for i in (2, 5, 9, 13, 17)]  # thumb MCP, index..pinky MCP
    return np.stack([_cos(dirs[i], dirs[i + 1]) for i in range(4)], axis=-1)


def tip_distances(n: np.ndarray) -> np.ndarray:
    """Pairwise distances between the five fingertips (10)."""
    return np.stack(
        [np.linalg.norm(n[..., a, :] - n[..., b, :], axis=-1) for a, b in combinations(TIPS, 2)],
        axis=-1,
    )


def tip_wrist_distances(n: np.ndarray) -> np.ndarray:
    """Distance of every fingertip to the wrist (5)."""
    return np.stack([np.linalg.norm(n[..., t, :], axis=-1) for t in TIPS], axis=-1)


def raw_features(lm: np.ndarray) -> np.ndarray:
    """Normalised coordinates of joints 1..20 (the wrist is always 0) -> 60 values."""
    n = normalize_landmarks(lm)
    return n[..., 1:, :].reshape(*n.shape[:-2], 60).astype(np.float32)


def engineered_features(lm: np.ndarray) -> np.ndarray:
    """Raw coordinates + joint angles + spread angles + tip distances -> 94 values."""
    n = normalize_landmarks(lm)
    parts = [
        n[..., 1:, :].reshape(*n.shape[:-2], 60),
        joint_angles(n),
        spread_angles(n),
        tip_distances(n),
        tip_wrist_distances(n),
    ]
    return np.concatenate(parts, axis=-1).astype(np.float32)


def graph_features(lm: np.ndarray) -> np.ndarray:
    """Node features for the graph network: (..., 21, 6) = coords + bone vector to parent."""
    n = normalize_landmarks(lm)
    bone = n - n[..., PARENT, :]
    return np.concatenate([n, bone], axis=-1).astype(np.float32)


def adjacency_matrix(self_loops: bool = True, normalize: bool = True) -> np.ndarray:
    """Adjacency of the hand skeleton, symmetric-normalised: D^-1/2 (A + I) D^-1/2."""
    a = np.zeros((NUM_LANDMARKS, NUM_LANDMARKS), dtype=np.float32)
    for i, j in HAND_CONNECTIONS:
        a[i, j] = a[j, i] = 1.0
    if self_loops:
        a += np.eye(NUM_LANDMARKS, dtype=np.float32)
    if normalize:
        d = a.sum(1)
        inv = 1.0 / np.sqrt(d)
        a = a * inv[:, None] * inv[None, :]
    return a.astype(np.float32)


FEATURE_BUILDERS = {
    "raw": raw_features,
    "engineered": engineered_features,
    "graph": graph_features,
}
