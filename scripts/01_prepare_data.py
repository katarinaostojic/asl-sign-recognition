"""Step 1: index the dataset, split it, cache resized images, and plot examples.

Outputs
  data/splits.csv                   one row per image with its split (train/val/test)
  data/cache/images_<size>.npy      resized uint8 images (memory-mapped later)
  results/figures/examples.png      a few examples per class
  results/figures/class_balance.png images per class and split
"""
import argparse
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aslrec import config, data, splits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="folder that contains the class folders A..Z, del, nothing, space")
    ap.add_argument("--size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--split", choices=["block", "random"], default="block")
    ap.add_argument("--block-size", type=int, default=config.BLOCK_SIZE)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    args = ap.parse_args()

    train_dir = config.find_train_dir(args.dataset)
    print("Dataset:", train_dir)
    df = data.build_index(train_dir)
    print(f"{len(df)} images in {df['label'].nunique()} classes")

    if args.split == "block":
        df["split"] = splits.block_split(df, args.block_size, config.SPLIT_FRACTIONS, args.seed)
    else:
        df["split"] = splits.random_split(df, config.SPLIT_FRACTIONS, args.seed)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.SPLITS_CSV, index=False)
    print(df["split"].value_counts().to_string())

    data.build_image_cache(df["path"].tolist(), args.size, data.cache_path(args.size), args.workers)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_class_balance(df)
    plot_examples(df, data.load_images(args.size))
    print("Done. Next: python scripts/02_extract_landmarks.py")


def plot_class_balance(df):
    counts = df.groupby(["class_name", "split"]).size().unstack(fill_value=0)
    counts = counts.reindex(config.CLASS_NAMES)[["train", "val", "test"]]
    ax = counts.plot(kind="bar", stacked=True, figsize=(12, 4))
    ax.set_ylabel("images")
    ax.set_title("Images per class and split")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close()


def plot_examples(df, images, per_class=6, seed=0):
    rng = np.random.default_rng(seed)
    n = len(config.CLASS_NAMES)
    fig, axes = plt.subplots(n // 3 + (n % 3 > 0), 3 * per_class, figsize=(3 * per_class * 0.9, n / 3 * 1.0 + 1))
    axes = np.atleast_2d(axes)
    for a in axes.ravel():
        a.axis("off")
    for c, name in enumerate(config.CLASS_NAMES):
        rows = np.flatnonzero(df["label"].to_numpy() == c)
        pick = rng.choice(rows, size=min(per_class, len(rows)), replace=False)
        r, col0 = divmod(c, 3)
        for k, row in enumerate(pick):
            ax = axes[r, col0 * per_class + k]
            ax.imshow(images[row])
            if k == 0:
                ax.set_title(name, fontsize=8, loc="left")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "examples.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
