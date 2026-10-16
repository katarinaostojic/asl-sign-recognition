"""Small image helpers shared by training, evaluation and the demos (OpenCV + NumPy)."""
from __future__ import annotations

import cv2
import numpy as np


def read_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize to size x size. The same function is used for the training cache and inference."""
    h, w = rgb.shape[:2]
    interp = cv2.INTER_AREA if max(h, w) > size else cv2.INTER_LINEAR
    return cv2.resize(rgb, (size, size), interpolation=interp)


def square_crop_around(rgb: np.ndarray, landmarks: np.ndarray, margin: float = 0.35) -> np.ndarray:
    """Crop a square around the hand (landmarks are normalised x, y, z) with a relative margin.

    The ASL Alphabet images are close-ups of the hand, so for photos/frames that show a
    whole scene we crop to the hand to approximate that framing before feeding the CNN.
    """
    h, w = rgb.shape[:2]
    xs, ys = landmarks[:, 0] * w, landmarks[:, 1] * h
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 * (1 + 2 * margin)
    half = max(half, 16)
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        rgb = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, x1, y0, y1 = x0 + pad, x1 + pad, y0 + pad, y1 + pad
    return rgb[y0:y1, x0:x1]
