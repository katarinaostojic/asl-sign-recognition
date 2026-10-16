"""Shared training logic for the Keras models (landmark nets and image nets)."""
from __future__ import annotations

import argparse
import time

import keras
import numpy as np

from . import config, data, features, tracking


def add_common_args(ap: argparse.ArgumentParser, epochs=60, batch=256, lr=1e-3, patience=8):
    ap.add_argument("--name", help="run name / model file name (default depends on the script)")
    ap.add_argument("--epochs", type=int, default=epochs)
    ap.add_argument("--batch-size", type=int, default=batch)
    ap.add_argument("--lr", type=float, default=lr)
    ap.add_argument("--patience", type=int, default=patience, help="early-stopping patience")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: tiny subset, 2 epochs (results are NOT meaningful)")


def set_seed(seed: int) -> None:
    keras.utils.set_random_seed(seed)


def make_callbacks(name: str, patience: int, monitor="val_accuracy"):
    run_dir = config.RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(monitor, patience=patience, restore_best_weights=True, mode="max"),
        keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=max(2, patience // 2),
                                          min_lr=1e-6),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
    ]


def compile_model(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def finish(name: str, model: keras.Model, params: dict, df, test_rows, probs, detected,
           history, t_train: float) -> dict:
    """Save the model, test predictions and a metrics summary."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / f"{name}.keras"
    model.save(path)
    y_true = df["label"].to_numpy()[test_rows]
    tracking.save_predictions(name, test_rows, y_true, probs, detected)

    n_det = int(detected.sum())
    acc_detected = float((probs[detected].argmax(1) == y_true[detected]).mean()) if n_det else float("nan")
    hist = history.history if history is not None else {}
    metrics = {
        "test_accuracy_on_detected": acc_detected,
        "test_detected_fraction": float(detected.mean()),
        "best_val_accuracy": float(max(hist.get("val_accuracy", [float("nan")]))),
        "epochs_run": len(hist.get("loss", [])),
        "train_seconds": t_train,
        "parameters": int(model.count_params()),
        "model_file_mb": path.stat().st_size / 1e6,
    }
    tracking.save_run(name, params, metrics)
    print(f"\n[{name}] test accuracy on detected images: {acc_detected:.4f} "
          f"(detected {metrics['test_detected_fraction']:.1%} of test images)")
    print(f"[{name}] saved model -> {path}")
    print("Next: python scripts/09_evaluate.py (after all models are trained)")
    return metrics


# ------------------------------------------------------------------ landmark networks --------
def train_landmark_net(name: str, build_fn, feature_mode: str, args, extra_params: dict | None = None):
    """Train a Keras model on landmark features. `build_fn(normalizer)` returns the model."""
    set_seed(args.seed)
    df, X, detected = data.load_landmark_arrays(feature_mode)
    y = df["label"].to_numpy()
    tr = data.split_rows(df, "train")
    va = data.split_rows(df, "val")
    te = data.split_rows(df, "test")
    if args.quick:
        tr, va, te = (data.subsample_rows(df, r, n) for r, n in ((tr, 300), (va, 60), (te, 60)))
        args.epochs = 2
    tr, va = tr[detected[tr]], va[detected[va]]  # train / tune only where a hand was found
    print(f"train {len(tr)}  val {len(va)}  test {len(te)} (all, incl. undetected)")

    norm = keras.layers.Normalization(axis=-1)
    norm.adapt(X[tr])
    model = build_fn(norm)
    model.summary(print_fn=lambda s: None)
    compile_model(model, args.lr)

    t0 = time.time()
    hist = model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=args.epochs,
                     batch_size=args.batch_size, callbacks=make_callbacks(name, args.patience),
                     verbose=2)
    t_train = time.time() - t0

    probs = np.zeros((len(te), config.NUM_CLASSES), dtype=np.float32)
    det_te = detected[te]
    if det_te.any():
        probs[det_te] = model.predict(X[te][det_te], batch_size=1024, verbose=0)
    params = {"model": name, "features": feature_mode, **vars(args), **(extra_params or {})}
    return finish(name, model, params, df, te, probs, det_te, hist, t_train)


# ------------------------------------------------------------------ image networks -----------
def image_datasets(args, batch_size):
    import tensorflow as tf

    from .models import make_augmenter

    df = data.load_index()
    images = data.load_images()
    y = df["label"].to_numpy()
    tr = data.split_rows(df, "train")
    va = data.split_rows(df, "val")
    te = data.split_rows(df, "test")
    if args.quick:
        tr, va, te = (data.subsample_rows(df, r, n) for r, n in ((tr, 300), (va, 60), (te, 60)))
        args.epochs = 2
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}")

    aug = make_augmenter()
    train_ds = data.batch_dataset(images, y, tr, batch_size, shuffle=True, seed=args.seed)
    train_ds = train_ds.map(lambda x, t: (aug(x, training=True), t), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = data.batch_dataset(images, y, va, batch_size, shuffle=False)
    test_ds = data.batch_dataset(images, y, te, batch_size, shuffle=False)
    return df, train_ds, val_ds, test_ds, te


def train_image_net(name: str, model: keras.Model, args, extra_params: dict | None = None,
                    finetune_fn=None, finetune_epochs: int = 0, finetune_lr: float = 1e-4):
    set_seed(args.seed)
    df, train_ds, val_ds, test_ds, te = image_datasets(args, args.batch_size)
    compile_model(model, args.lr)
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                     callbacks=make_callbacks(name, args.patience), verbose=2)
    if finetune_fn is not None and finetune_epochs > 0:
        print("\n--- fine-tuning the backbone ---")
        finetune_fn(model)
        compile_model(model, finetune_lr)
        ft = model.fit(train_ds, validation_data=val_ds, epochs=finetune_epochs,
                       callbacks=make_callbacks(name + "_ft", max(3, args.patience // 2)), verbose=2)
        for k, v in ft.history.items():  # merge for the summary
            hist.history.setdefault(k, []).extend(v)
    t_train = time.time() - t0

    probs = model.predict(test_ds, verbose=0)
    detected = np.ones(len(te), dtype=bool)
    params = {"model": name, **vars(args), **(extra_params or {})}
    return finish(name, model, params, df, te, probs, detected, hist, t_train)
