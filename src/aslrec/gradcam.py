"""Grad-CAM for the image models (custom CNN and MobileNetV2).

The image models are plain chains of layers, so we run the layers one by one inside a
GradientTape, remember the last convolutional feature map and differentiate the chosen
class score with respect to it.
"""
from __future__ import annotations

import cv2
import keras
import numpy as np
import tensorflow as tf


def _forward(model: keras.Model, x: tf.Tensor, conv_layer: str):
    """Return (last conv feature map, predictions) running the chain layer by layer."""
    h, conv = x, None
    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        if isinstance(layer, keras.Model):  # nested backbone (MobileNetV2)
            inner = keras.Model(layer.input, [layer.get_layer("out_relu").output, layer.output])
            conv, h = inner(h, training=False)
        else:
            h = layer(h, training=False)
            if layer.name == conv_layer:
                conv = h
    return conv, h


def grad_cam(model: keras.Model, image: np.ndarray, class_index: int | None = None,
             conv_layer: str = "last_conv") -> tuple[np.ndarray, int, float]:
    """Heat-map in [0, 1] with the size of `image` (H, W), the explained class and its probability.

    `image` is a single (H, W, 3) array in [0, 255]. If class_index is None the predicted class is used.
    """
    x = tf.convert_to_tensor(image[None].astype(np.float32))
    with tf.GradientTape() as tape:
        tape.watch(x)
        conv, preds = _forward(model, x, conv_layer)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        score = preds[:, class_index]
    grads = tape.gradient(score, conv)                       # (1, h, w, c)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv, axis=-1))[0].numpy()
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() > 1e-8 else cam
    return cam, class_index, float(preds[0, class_index])


def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a jet heat-map onto an RGB uint8 image."""
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (image.astype(np.float32) * (1 - alpha) + heat * alpha).astype(np.uint8)


def hand_focus(cam: np.ndarray, landmarks: np.ndarray, margin: float = 0.15) -> float:
    """Share of the heat-map mass that lies inside the hand's (padded) bounding box.

    `landmarks` are MediaPipe landmarks normalised to the image. A value near 1 means the
    model looks at the hand; the area of the box itself is a useful reference (a uniform
    map would give area_box / area_image).
    """
    h, w = cam.shape
    xs, ys = landmarks[:, 0] * w, landmarks[:, 1] * h
    dx, dy = (xs.max() - xs.min()) * margin, (ys.max() - ys.min()) * margin
    x0, x1 = int(max(xs.min() - dx, 0)), int(min(xs.max() + dx, w))
    y0, y1 = int(max(ys.min() - dy, 0)), int(min(ys.max() + dy, h))
    total = cam.sum()
    if total <= 1e-8 or x1 <= x0 or y1 <= y0:
        return float("nan")
    return float(cam[y0:y1, x0:x1].sum() / total)
