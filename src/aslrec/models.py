"""Keras model definitions: MLP, hand-skeleton graph network, CNN and MobileNetV2.

All image models take float32 images in [0, 255] of shape (size, size, 3) and rescale
internally, so the exact same graph can be exported to TFLite. Data augmentation is
NOT part of the models; it is applied in the tf.data pipeline (see `make_augmenter`).
"""
from __future__ import annotations

import keras
from keras import layers

from . import config, features


# ------------------------------------------------------------------ landmark models ----------
def build_mlp(input_dim: int, normalizer: layers.Normalization | None = None,
              hidden=(256, 128), dropout=0.3, activation="relu", noise=0.02, l2=1e-4,
              name="mlp") -> keras.Model:
    reg = keras.regularizers.l2(l2) if l2 else None
    inp = keras.Input((input_dim,), name="features")
    x = normalizer(inp) if normalizer is not None else inp
    x = layers.GaussianNoise(noise)(x)  # only active during training
    for h in hidden:
        x = layers.Dense(h, kernel_regularizer=reg, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(config.NUM_CLASSES, activation="softmax", name="probs")(x)
    return keras.Model(inp, out, name=name)


@keras.saving.register_keras_serializable(package="aslrec")
class GraphConv(layers.Layer):
    """Graph convolution over the fixed hand skeleton: act(A_hat @ X @ W + b)."""

    def __init__(self, units: int, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.adj = features.adjacency_matrix()  # (21, 21) constant, symmetric-normalised

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="glorot_uniform",
                                 name="w")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", name="b")

    def call(self, x):
        h = keras.ops.matmul(x, self.w)                       # (B, 21, units)
        h = keras.ops.matmul(keras.ops.convert_to_tensor(self.adj), h)  # aggregate neighbours
        return self.activation(h + self.b)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.units)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(units=self.units, activation=keras.activations.serialize(self.activation))
        return cfg


def build_gnn(normalizer: layers.Normalization | None = None, hidden=(64, 64, 32), dropout=0.3,
              noise=0.02, name="gnn") -> keras.Model:
    inp = keras.Input((features.NUM_LANDMARKS, 6), name="nodes")
    x = normalizer(inp) if normalizer is not None else inp
    x = layers.GaussianNoise(noise)(x)
    for h in hidden:
        x = GraphConv(h)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.Flatten()(x)  # joints are not interchangeable, so keep their position
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(config.NUM_CLASSES, activation="softmax", name="probs")(x)
    return keras.Model(inp, out, name=name)


# ------------------------------------------------------------------ image models -------------
def make_augmenter(rotation=0.08, translation=0.10, zoom=0.15, brightness=0.25, contrast=0.25):
    """Random augmentation for images in [0, 255]; used inside tf.data (training only)."""
    return keras.Sequential(
        [
            layers.RandomRotation(rotation, fill_mode="reflect"),
            layers.RandomTranslation(translation, translation, fill_mode="reflect"),
            layers.RandomZoom((-zoom, zoom), fill_mode="reflect"),
            layers.RandomBrightness(brightness, value_range=(0, 255)),
            layers.RandomContrast(contrast, value_range=(0, 255)),
        ],
        name="augmentation",
    )


def build_cnn(size=config.IMG_SIZE, blocks=4, base_filters=32, kernel=3, activation="relu",
              dropout=0.4, dense=128, name="cnn") -> keras.Model:
    """CNN trained from scratch: `blocks` x (2 x [conv-BN-act]) + pooling, GAP, dense head."""
    inp = keras.Input((size, size, 3), name="image")
    x = layers.Rescaling(1.0 / 255)(inp)
    for i in range(blocks):
        f = base_filters * 2 ** min(i, 3)
        x = layers.Conv2D(f, kernel, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Conv2D(f, kernel, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation, name="last_conv" if i == blocks - 1 else None)(x)
        if i < blocks - 1:
            x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense, activation=activation)(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(config.NUM_CLASSES, activation="softmax", name="probs")(x)
    return keras.Model(inp, out, name=name)


def build_mobilenet(size=config.IMG_SIZE, dropout=0.3, weights="imagenet", name="mobilenet") -> keras.Model:
    """MobileNetV2 backbone (ImageNet weights) + new classification head."""
    base = keras.applications.MobileNetV2(
        input_shape=(size, size, 3), include_top=False, weights=weights, alpha=1.0
    )
    base.trainable = False
    inp = keras.Input((size, size, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inp)  # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(config.NUM_CLASSES, activation="softmax", name="probs")(x)
    return keras.Model(inp, out, name=name)


def unfreeze_backbone(model: keras.Model, n_trainable_layers: int = 40) -> None:
    """Fine-tuning: make the last `n_trainable_layers` of the backbone trainable.

    BatchNorm layers stay in inference mode because the backbone is called with training=False.
    """
    base = next(l for l in model.layers if isinstance(l, keras.Model))
    base.trainable = True
    for layer in base.layers[:-n_trainable_layers]:
        layer.trainable = False
