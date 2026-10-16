"""Image perturbations that mimic a domain shift (lighting, blur, noise, camera distance, ...).

The ASL Alphabet images were all recorded in a similar setting, so a high test accuracy says
little about real-world use. These perturbations give a cheap, reproducible stress test
that needs no extra data. Every function maps an RGB uint8 image to an RGB uint8 image.
"""
from __future__ import annotations

import cv2
import numpy as np


def identity(img, rng):
    return img


def darker(img, rng):
    return np.clip(img.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)


def brighter(img, rng):
    return np.clip(img.astype(np.float32) * 1.6 + 25, 0, 255).astype(np.uint8)


def low_contrast(img, rng):
    return np.clip((img.astype(np.float32) - 128) * 0.5 + 128, 0, 255).astype(np.uint8)


def blur(img, rng):
    return cv2.GaussianBlur(img, (0, 0), 2.5)


def noise(img, rng):
    return np.clip(img.astype(np.float32) + rng.normal(0, 18, img.shape), 0, 255).astype(np.uint8)


def jpeg(img, rng):
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 12])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _affine(img, angle=0.0, scale=1.0, tx=0.0, ty=0.0, rng=None):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += (tx * w, ty * h)
    border = tuple(int(v) for v in img.reshape(-1, 3).mean(0))
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def rotate(img, rng):
    return _affine(img, angle=float(rng.choice([-20, 20])))


def farther(img, rng):
    """Hand appears smaller (camera further away): shrink and pad with a mean-colour border."""
    return _affine(img, scale=0.6)


def shifted(img, rng):
    return _affine(img, tx=0.2 * float(rng.choice([-1, 1])), ty=0.15 * float(rng.choice([-1, 1])))


def colour_cast(img, rng):
    gain = np.array([1.25, 1.0, 0.7], dtype=np.float32)  # warm light
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def clutter(img, rng):
    """Random coloured rectangles in the corners/edges (busy background, partial occlusion)."""
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(4):
        rw, rh = int(w * rng.uniform(0.1, 0.25)), int(h * rng.uniform(0.1, 0.25))
        x = int(rng.choice([0, w - rw])); y = int(rng.integers(0, h - rh))
        out[y : y + rh, x : x + rw] = rng.integers(0, 255, 3)
    return out


PERTURBATIONS = {
    "none": identity, "dark": darker, "bright": brighter, "low_contrast": low_contrast,
    "blur": blur, "noise": noise, "jpeg_q12": jpeg, "rotate_20deg": rotate,
    "farther_away": farther, "shifted": shifted, "warm_light": colour_cast, "clutter": clutter,
}
