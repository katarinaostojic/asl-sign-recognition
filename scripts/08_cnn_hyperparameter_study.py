"""Step 8: effect of CNN hyper-parameters (depth, kernel size, activation, dropout).

Trains a small grid of CNNs for a few epochs on a subset of the training data and records
the validation accuracy of each. Vary one factor at a time around the default configuration.
This uses the validation split only; the test split is never touched.

Output: results/cnn_hyperparameter_study.csv and results/figures/cnn_hyperparameter_study.png
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import keras
import matplotlib.pyplot as plt
import pandas as pd

from aslrec import config, data, models, training

BASE = dict(blocks=4, kernel=3, activation="relu", dropout=0.4)
VARIANTS = (
    [dict(BASE, blocks=b) for b in (2, 3, 4, 5)]
    + [dict(BASE, kernel=k) for k in (3, 5, 7)]
    + [dict(BASE, activation=a) for a in ("relu", "elu", "swish")]
    + [dict(BASE, dropout=d) for d in (0.0, 0.2, 0.4, 0.6)]
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--per-class", type=int, default=400, help="training images per class")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.per_class = 1, 30

    import tensorflow as tf

    df = data.load_index()
    images = data.load_images()
    y = df["label"].to_numpy()
    tr = data.subsample_rows(df, data.split_rows(df, "train"), args.per_class, args.seed)
    va = data.subsample_rows(df, data.split_rows(df, "val"), max(40, args.per_class // 4), args.seed)
    aug = models.make_augmenter()

    seen, rows = set(), []
    for cfg in VARIANTS:
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        training.set_seed(args.seed)
        train_ds = data.batch_dataset(images, y, tr, args.batch_size, True, args.seed).map(
            lambda x, t: (aug(x, training=True), t), num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = data.batch_dataset(images, y, va, args.batch_size, False)
        model = models.build_cnn(config.IMG_SIZE, cfg["blocks"], 32, cfg["kernel"], cfg["activation"],
                                 cfg["dropout"])
        training.compile_model(model, 1e-3)
        t0 = time.time()
        h = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=0)
        row = dict(cfg, val_accuracy=max(h.history["val_accuracy"]), params=model.count_params(),
                   seconds=round(time.time() - t0, 1))
        rows.append(row)
        print(row)
        keras.backend.clear_session()

    res = pd.DataFrame(rows)
    res.to_csv(config.RESULTS_DIR / "cnn_hyperparameter_study.csv", index=False)

    factors = ["blocks", "kernel", "activation", "dropout"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    for ax, f in zip(axes, factors):
        others = {k: v for k, v in BASE.items() if k != f}
        sub = res[(res[list(others)] == pd.Series(others)).all(axis=1)].sort_values(f)
        ax.plot(sub[f].astype(str), sub["val_accuracy"], "o-")
        ax.set_title(f)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("validation accuracy")
    plt.suptitle(f"CNN hyper-parameter study ({args.epochs} epochs, {args.per_class} img/class)")
    plt.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIGURES_DIR / "cnn_hyperparameter_study.png", dpi=150)


if __name__ == "__main__":
    main()
