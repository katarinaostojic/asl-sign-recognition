"""MediaPipe hand-landmark extraction (Tasks API, works with current MediaPipe releases)."""
from __future__ import annotations

import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import config
from .features import HAND_CONNECTIONS
from .imaging import read_rgb

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kw):
        return x


def ensure_hand_model(path: Path = config.HAND_MODEL_PATH) -> Path:
    """Download the MediaPipe hand_landmarker.task model once."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe hand model to {path} ...")
        urllib.request.urlretrieve(config.HAND_MODEL_URL, path)
    return path


class HandDetector:
    """Wrapper returning 21 (x, y, z) landmarks in coordinates of the *input* image.

    The ASL Alphabet photos are tight close-ups, which the palm detector handles poorly,
    so by default the image is padded with a gray border (and upscaled) before detection.
    x and y are normalised to the original image (values slightly outside [0, 1] are possible).
    """

    def __init__(self, model_path: Path | None = None, pad: float = 0.5, upscale: int = 2,
                 min_conf: float = 0.3):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        model_path = ensure_hand_model(model_path or config.HAND_MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self._det = vision.HandLandmarker.create_from_options(options)
        self.pad, self.upscale = pad, upscale

    def _run(self, rgb: np.ndarray):
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._det.detect(img)
        if not res.hand_landmarks:
            return None
        return np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]], dtype=np.float32)

    def detect(self, rgb: np.ndarray) -> np.ndarray | None:
        """Return (21, 3) landmarks or None when no hand is found."""
        h, w = rgb.shape[:2]
        if self.pad > 0:
            p = int(round(max(h, w) * self.pad))
            padded = cv2.copyMakeBorder(rgb, p, p, p, p, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            if self.upscale > 1 and max(padded.shape[:2]) < 640:
                padded = cv2.resize(padded, None, fx=self.upscale, fy=self.upscale,
                                    interpolation=cv2.INTER_LINEAR)
            lm = self._run(padded)
            if lm is not None:
                # landmarks are normalised to the padded image; map back to the original one
                lm[:, 0] = (lm[:, 0] * (w + 2 * p) - p) / w
                lm[:, 1] = (lm[:, 1] * (h + 2 * p) - p) / h
                return lm
        return self._run(rgb)

    def close(self) -> None:
        self._det.close()


# ---- parallel extraction over many files -------------------------------------------------

_worker_detector: HandDetector | None = None


def _init_worker(model_path: str) -> None:
    global _worker_detector
    _worker_detector = HandDetector(Path(model_path))


def _detect_path(path: str):
    lm = _worker_detector.detect(read_rgb(path))
    return lm


def extract_from_paths(paths: list[str], workers: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Run MediaPipe on many image files. Returns landmarks (N, 21, 3) and detected (N,)."""
    model = ensure_hand_model()  # download once in the parent process
    n = len(paths)
    lms = np.zeros((n, 21, 3), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    if workers <= 1:
        _init_worker(str(model))
        results = map(_detect_path, paths)
        for i, lm in enumerate(tqdm(results, total=n, desc="landmarks")):
            if lm is not None:
                lms[i], ok[i] = lm, True
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(str(model),)) as ex:
            for i, lm in enumerate(tqdm(ex.map(_detect_path, paths, chunksize=64), total=n,
                                        desc="landmarks")):
                if lm is not None:
                    lms[i], ok[i] = lm, True
    return lms, ok


def draw_landmarks(rgb: np.ndarray, lm: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    """Return a copy of `rgb` with the hand skeleton drawn on it."""
    out = np.ascontiguousarray(rgb.copy())
    h, w = out.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(out, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(out, p, 3, (255, 0, 0), -1)
    return out


def time_detection(detector: HandDetector, images: list[np.ndarray]) -> np.ndarray:
    """Per-image detection time in milliseconds."""
    times = []
    for img in images:
        t = time.perf_counter()
        detector.detect(img)
        times.append((time.perf_counter() - t) * 1000)
    return np.array(times)
