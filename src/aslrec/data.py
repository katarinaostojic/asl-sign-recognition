"""Dataset indexing, image cache and tf.data input pipeline."""
from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, features
from .imaging import read_rgb, resize_rgb

try:  # progress bars are optional
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kw):
        return x


def build_index(train_dir: Path) -> pd.DataFrame:
    """List every training image with its class label and frame number."""
    rows = []
    for label, cname in enumerate(config.CLASS_NAMES):
        d = Path(train_dir) / cname
        if not d.is_dir():
            raise FileNotFoundError(f"Missing class folder: {d}")
        for p in d.iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                m = re.search(r"(\d+)$", p.stem)
                rows.append((str(p), cname, label, int(m.group(1)) if m else -1))
    df = pd.DataFrame(rows, columns=["path", "class_name", "label", "idx"])
    return df.sort_values(["label", "idx", "path"]).reset_index(drop=True)


def load_index() -> pd.DataFrame:
    if not config.SPLITS_CSV.exists():
        raise FileNotFoundError("data/splits.csv not found. Run scripts/01_prepare_data.py first.")
    return pd.read_csv(config.SPLITS_CSV)


def split_rows(df: pd.DataFrame, split: str) -> np.ndarray:
    return np.flatnonzero(df["split"].to_numpy() == split)


def _load_resize(path: str, size: int) -> np.ndarray:
    return resize_rgb(read_rgb(path), size)


def build_image_cache(paths: list[str], size: int, out_path: Path, workers: int = 4) -> None:
    """Resize all images once and store them as one uint8 array (N, size, size, 3)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.uint8, shape=(len(paths), size, size, 3)
    )
    fn = partial(_load_resize, size=size)
    if workers <= 1:
        for i, p in enumerate(tqdm(paths, desc="resize")):
            arr[i] = fn(p)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, img in enumerate(tqdm(ex.map(fn, paths, chunksize=256), total=len(paths), desc="resize")):
                arr[i] = img
    arr.flush()


def cache_path(size: int = config.IMG_SIZE) -> Path:
    return config.CACHE_DIR / f"images_{size}.npy"


def load_images(size: int = config.IMG_SIZE) -> np.ndarray:
    p = cache_path(size)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run scripts/01_prepare_data.py first.")
    return np.load(p, mmap_mode="r")


def batch_dataset(images, labels, rows, batch_size, shuffle, seed=0):
    """tf.data pipeline that reads batches from the (memory-mapped) uint8 cache.

    Yields float32 images in [0, 255] (the models rescale internally) and int labels.
    Random augmentation lives inside the model, so the cache never changes.
    """
    import tensorflow as tf

    rows = np.asarray(rows)
    rng = np.random.default_rng(seed)
    size = images.shape[1]

    def gen():
        order = rng.permutation(rows) if shuffle else rows
        for s in range(0, len(order), batch_size):
            b = np.sort(order[s : s + batch_size])
            yield images[b].astype(np.float32), labels[b].astype(np.int32)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((None, size, size, 3), tf.float32),
            tf.TensorSpec((None,), tf.int32),
        ),
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def subsample_rows(df: pd.DataFrame, rows: np.ndarray, per_class: int, seed: int = 0) -> np.ndarray:
    """Keep at most `per_class` rows of each class (used by --quick smoke runs)."""
    rng = np.random.default_rng(seed)
    keep = []
    labels = df["label"].to_numpy()
    for c in np.unique(labels[rows]):
        r = rows[labels[rows] == c]
        keep.append(rng.choice(r, size=min(per_class, len(r)), replace=False))
    return np.sort(np.concatenate(keep))


def load_landmark_arrays(feature_mode: str):
    df = load_index()
    if not config.LANDMARKS_NPZ.exists():
        raise FileNotFoundError("Run scripts/02_extract_landmarks.py first.")
    z = np.load(config.LANDMARKS_NPZ)
    lm, detected = z["landmarks"], z["detected"]
    if len(lm) != len(df):
        raise ValueError("landmarks.npz and splits.csv are out of sync; re-run steps 1 and 2.")
    X = features.FEATURE_BUILDERS[feature_mode](lm)
    return df, X, detected
