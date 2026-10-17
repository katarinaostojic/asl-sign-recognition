"""Load trained models (Keras / scikit-learn / TFLite) behind one `predict_proba` interface,
and wrap them in end-to-end pipelines that start from an RGB image."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config, features
from .imaging import resize_rgb, square_crop_around

# name -> how the model consumes data. Extra names (e.g. mlp_raw) are resolved by prefix.
MODEL_KINDS = {
    "knn": ("landmark", "engineered"),
    "knn_raw": ("landmark", "raw"),
    "mlp": ("landmark", "engineered"),
    "mlp_raw": ("landmark", "raw"),
    "gnn": ("landmark", "graph"),
    "cnn": ("image", None),
    "mobilenet": ("image", None),
}


def describe(name: str) -> tuple[str, str | None]:
    base = name.split("_tflite")[0].split("_fp16")[0].split("_int8")[0].split("_dynamic")[0]
    if base in MODEL_KINDS:
        return MODEL_KINDS[base]
    for key in sorted(MODEL_KINDS, key=len, reverse=True):
        if base.startswith(key):
            return MODEL_KINDS[key]
    raise KeyError(f"Unknown model name: {name}")


class SklearnPredictor:
    def __init__(self, path):
        import joblib

        self.model = joblib.load(path)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), config.NUM_CLASSES), dtype=np.float32)
        out[:, self.model.classes_] = self.model.predict_proba(x)
        return out


class KerasPredictor:
    def __init__(self, path):
        import keras

        from . import models  # noqa: F401  (registers the GraphConv layer)

        import tensorflow as tf

        self.model = keras.models.load_model(path, compile=False)
        # a compiled graph avoids the large per-call overhead of eager execution, which would
        # otherwise exaggerate the speed-up of TFLite in latency comparisons
        self._fn = tf.function(lambda t: self.model(t, training=False), reduce_retracing=True)
        self._tf = tf

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if len(x) <= 8:
            return self._fn(self._tf.convert_to_tensor(x)).numpy()
        return self.model.predict(x, batch_size=256, verbose=0)


class TFLitePredictor:
    """Runs a .tflite model sample by sample (batch size 1)."""

    def __init__(self, path, num_threads: int = 1):
        Interpreter = None
        for mod in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter"):
            try:
                Interpreter = __import__(mod, fromlist=["Interpreter"]).Interpreter
                break
            except ImportError:
                continue
        if Interpreter is None:
            import tensorflow as tf

            Interpreter = tf.lite.Interpreter
        self.interp = Interpreter(model_path=str(path), num_threads=num_threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        res = []
        for sample in x:
            self.interp.set_tensor(self.inp["index"], sample[None].astype(self.inp["dtype"]))
            self.interp.invoke()
            res.append(self.interp.get_tensor(self.out["index"])[0])
        return np.stack(res).astype(np.float32)


def load_predictor(path):
    path = Path(path)
    if path.suffix == ".joblib":
        return SklearnPredictor(path)
    if path.suffix == ".keras":
        return KerasPredictor(path)
    if path.suffix == ".tflite":
        return TFLitePredictor(path)
    raise ValueError(f"Unsupported model file: {path}")


def find_model_file(name: str) -> Path:
    for p in (config.MODELS_DIR / f"{name}.joblib", config.MODELS_DIR / f"{name}.keras",
              config.MODELS_DIR / "tflite" / f"{name}.tflite"):
        if p.exists():
            return p
    raise FileNotFoundError(f"No trained model named '{name}' in {config.MODELS_DIR}")


class LandmarkPipeline:
    """RGB image -> MediaPipe -> features -> model -> class probabilities.

    If no hand is detected the pipeline answers 'nothing' with probability 1; this is the
    "end-to-end" behaviour the evaluation reports.
    """

    kind = "landmark"

    def __init__(self, predictor, feature_mode: str, detector=None):
        self.predictor = predictor
        self.build = features.FEATURE_BUILDERS[feature_mode]
        if detector is None:
            from .landmarks import HandDetector

            detector = HandDetector()
        self.detector = detector

    def predict_with_landmarks(self, rgb: np.ndarray, lm: np.ndarray | None) -> np.ndarray:
        """Probabilities (29,) for one frame whose landmarks were already detected."""
        if lm is None:
            p = np.zeros(config.NUM_CLASSES, dtype=np.float32)
            p[config.NOTHING_INDEX] = 1.0
            return p
        return self.predictor.predict_proba(self.build(lm[None]))[0]

    def predict_rgb(self, images: list[np.ndarray]):
        n = len(images)
        probs = np.zeros((n, config.NUM_CLASSES), dtype=np.float32)
        probs[:, config.NOTHING_INDEX] = 1.0
        lms = np.full((n, 21, 3), np.nan, dtype=np.float32)
        found = []
        for i, img in enumerate(images):
            lm = self.detector.detect(img)
            if lm is not None:
                lms[i] = lm
                found.append(i)
        if found:
            probs[found] = self.predictor.predict_proba(self.build(lms[found]))
        detected = np.zeros(n, dtype=bool)
        detected[found] = True
        return probs, detected, lms


class ImagePipeline:
    """RGB image -> (optional hand crop) -> resize -> CNN -> class probabilities."""

    kind = "image"

    def __init__(self, predictor, size: int = config.IMG_SIZE, crop_hand: bool = False, detector=None):
        self.predictor, self.size, self.crop_hand = predictor, size, crop_hand
        if crop_hand and detector is None:
            from .landmarks import HandDetector

            detector = HandDetector()
        self.detector = detector

    def predict_with_landmarks(self, rgb: np.ndarray, lm: np.ndarray | None) -> np.ndarray:
        if lm is None:
            p = np.zeros(config.NUM_CLASSES, dtype=np.float32)
            p[config.NOTHING_INDEX] = 1.0
            return p
        crop = square_crop_around(rgb, lm) if self.crop_hand else rgb
        return self.predictor.predict_proba(resize_rgb(crop, self.size)[None].astype(np.float32))[0]

    def predict_rgb(self, images: list[np.ndarray]):
        batch, lms = [], np.full((len(images), 21, 3), np.nan, dtype=np.float32)
        detected = np.ones(len(images), dtype=bool)
        for i, img in enumerate(images):
            if self.crop_hand:
                lm = self.detector.detect(img)
                if lm is not None:
                    lms[i] = lm
                    img = square_crop_around(img, lm)
                else:
                    detected[i] = False
            batch.append(resize_rgb(img, self.size))
        probs = self.predictor.predict_proba(np.stack(batch).astype(np.float32))
        return probs, detected, lms


def available_model_names() -> list[str]:
    """Names of trained models found in models/ (Keras, scikit-learn and TFLite files)."""
    names = [p.stem for p in config.MODELS_DIR.glob("*.keras")] + \
            [p.stem for p in config.MODELS_DIR.glob("*.joblib")]
    tfl = config.MODELS_DIR / "tflite"
    if tfl.exists():
        names += [p.stem for p in tfl.glob("*.tflite")]
    return sorted(names)


def build_pipeline(name: str, path=None, detector=None, crop_hand: bool = False):
    kind, feats = describe(name)
    predictor = load_predictor(path or find_model_file(name))
    if kind == "landmark":
        return LandmarkPipeline(predictor, feats, detector)
    return ImagePipeline(predictor, config.IMG_SIZE, crop_hand, detector)
