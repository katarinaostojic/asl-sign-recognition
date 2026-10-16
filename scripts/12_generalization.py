"""Step 12: does the accuracy survive a change of conditions?

Part A - synthetic domain shift: perturb test images (darker, blurred, noisy, smaller hand,
         rotated, cluttered, ...) and re-run each full pipeline (MediaPipe included).
Part B - your own photos: put images in data/custom/<CLASS>/*.jpg (CLASS = A..Z, del, nothing,
         space; capture them with tools/capture_photos.py) and evaluate every model on them.
         Image models are evaluated twice: on the whole photo and on a hand-centred crop
         (the dataset images are close-ups, so cropping imitates that framing).

Outputs: results/robustness.csv, results/custom_photos.csv, results/figures/robustness.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from aslrec import config, data, inference, robustness
from aslrec.imaging import read_rgb
from aslrec.landmarks import HandDetector


def available_models(requested):
    if requested:
        return requested
    return [n for n in inference.MODEL_KINDS
            if any((config.MODELS_DIR / f"{n}{s}").exists() for s in (".joblib", ".keras"))]


def part_a(names, detector, n, seed):
    df = data.load_index()
    te = data.split_rows(df, "test")
    rng = np.random.default_rng(seed)
    pick = data.subsample_rows(df, te, max(1, n // config.NUM_CLASSES), seed)
    imgs = [read_rgb(df["path"].iloc[i]) for i in pick]
    y = df["label"].to_numpy()[pick]
    pipes = {m: inference.build_pipeline(m, detector=detector) for m in names}
    rows = []
    for pname, fn in robustness.PERTURBATIONS.items():
        prng = np.random.default_rng(seed + 1)
        pert = [fn(im, prng) for im in imgs]
        for m, pipe in pipes.items():
            probs, det, _ = pipe.predict_rgb(pert)
            rows.append({"model": m, "perturbation": pname, "accuracy": float((probs.argmax(1) == y).mean()),
                         "detected": float(det.mean())})
        print(pname, {r["model"]: round(r["accuracy"], 3) for r in rows[-len(pipes):]})
    res = pd.DataFrame(rows)
    res.to_csv(config.RESULTS_DIR / "robustness.csv", index=False)
    piv = res.pivot(index="model", columns="perturbation", values="accuracy")[list(robustness.PERTURBATIONS)]
    plt.figure(figsize=(13, 0.6 * len(names) + 2))
    sns.heatmap(piv.loc[names], annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1)
    plt.title(f"Accuracy under perturbations ({len(imgs)} test images)")
    plt.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIGURES_DIR / "robustness.png", dpi=150)
    plt.close()


def part_b(names, detector):
    if not config.CUSTOM_DIR.exists():
        print("\nPart B skipped: no photos in data/custom/<CLASS>/ (see tools/capture_photos.py).")
        return
    paths, labels = [], []
    lower = {c.lower(): i for i, c in enumerate(config.CLASS_NAMES)}
    for d in sorted(config.CUSTOM_DIR.iterdir()):
        if d.is_dir() and d.name.lower() in lower:
            for p in d.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    paths.append(p); labels.append(lower[d.name.lower()])
    if not paths:
        print("\nPart B skipped: data/custom is empty.")
        return
    y = np.array(labels)
    imgs = [read_rgb(p) for p in paths]
    print(f"\nPart B: {len(imgs)} own photos in {len(set(labels))} classes")
    rows, per_class = [], []
    variants = []
    for m in names:
        kind, _ = inference.describe(m)
        variants.append((m, m, False))
        if kind == "image":
            variants.append((f"{m}+handcrop", m, True))
    for label, m, crop in variants:
        pipe = inference.build_pipeline(m, detector=detector, crop_hand=crop)
        probs, det, _ = pipe.predict_rgb(imgs)
        pred = probs.argmax(1)
        rows.append({"model": label, "accuracy": float((pred == y).mean()), "hand_detected": float(det.mean()),
                     "n": len(y)})
        for c in np.unique(y):
            per_class.append({"model": label, "class": config.CLASS_NAMES[c],
                              "accuracy": float((pred[y == c] == c).mean()), "n": int((y == c).sum())})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "custom_photos.csv", index=False)
    pd.DataFrame(per_class).to_csv(config.RESULTS_DIR / "custom_photos_per_class.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--n", type=int, default=870, help="test images used in part A (~30 per class)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    names = available_models(args.models)
    detector = HandDetector()
    part_a(names, detector, args.n, args.seed)
    part_b(names, detector)


if __name__ == "__main__":
    main()
