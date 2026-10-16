"""Train / validation / test splitting.

The ASL Alphabet images are consecutive video frames, so neighbouring images are
near-duplicates. A plain random split therefore leaks almost identical frames into
train and test and inflates accuracy. The default `block` split keeps runs of
consecutive frames together, so test frames come from different moments of the
recording than the training frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def _counts(n: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    n_train = int(round(n * fractions[0]))
    n_val = int(n * fractions[1])
    if n >= 3:
        n_val = max(n_val, 1)
        n_train = min(n_train, n - n_val - 1)  # keep at least one block for test
    n_test = n - n_train - n_val
    if n_train < 1 and n >= 1:  # tiny inputs (smoke tests)
        n_train, n_val, n_test = max(n - 2, 1), min(1, max(n - 1, 0)), (1 if n > 1 else 0)
    return n_train, n_val, n_test


def block_split(
    df: pd.DataFrame,
    block_size: int = 100,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> pd.Series:
    """Assign whole blocks of consecutive frames (per class) to train/val/test."""
    rng = np.random.default_rng(seed)
    out = pd.Series(index=df.index, dtype=object)
    for _, g in df.groupby("label"):
        g = g.sort_values("idx")
        block_id = np.arange(len(g)) // block_size
        blocks = np.unique(block_id)
        order = rng.permutation(blocks)
        n_train, n_val, _ = _counts(len(blocks), fractions)
        assign = {}
        for k, b in enumerate(order):
            assign[b] = "train" if k < n_train else ("val" if k < n_train + n_val else "test")
        out.loc[g.index] = [assign[b] for b in block_id]
    return out


def random_split(
    df: pd.DataFrame,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> pd.Series:
    """Stratified random split (leaky for this dataset; kept for comparison)."""
    rng = np.random.default_rng(seed)
    out = pd.Series(index=df.index, dtype=object)
    for _, g in df.groupby("label"):
        perm = rng.permutation(g.index.to_numpy())
        n_train, n_val, _ = _counts(len(perm), fractions)
        out.loc[perm[:n_train]] = "train"
        out.loc[perm[n_train : n_train + n_val]] = "val"
        out.loc[perm[n_train + n_val :]] = "test"
    return out
