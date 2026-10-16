"""Step 10: Grad-CAM for the image models.

Outputs (per model)
  results/figures/gradcam_<model>_correct.png   one correctly classified test image per class
  results/figures/gradcam_<model>_errors.png    most confident mistakes
  results/gradcam_hand_focus.csv                share of CAM mass inside the hand bounding box,
                                                compared with the box area (uniform-map baseline)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aslrec import config, data, gradcam, models  # noqa: F401  (models registers custom layers)

NAMES = config.CLASS_NAMES


def show_grid(items, title, path, ncols=8):
    n = len(items)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 2.05 * nrows), squeeze=False)
    for a in axes.ravel():
        a.axis("off")
    for a, (img, cam, caption) in zip(axes.ravel(), items):
        a.imshow(gradcam.overlay(img, cam))
        a.set_title(caption, fontsize=8)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["cnn", "mobilenet"])
    ap.add_argument("--focus-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = data.load_index()
    images = data.load_images()
    z = np.load(config.LANDMARKS_NPZ)
    lms, detected = z["landmarks"], z["detected"]
    rng = np.random.default_rng(args.seed)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    focus_rows = []

    for name in args.models:
        mpath = config.MODELS_DIR / f"{name}.keras"
        ppath = config.PRED_DIR / f"{name}.npz"
        if not mpath.exists() or not ppath.exists():
            print(f"skip {name}: model or predictions missing")
            continue
        model = keras.models.load_model(mpath, compile=False)
        p = np.load(ppath)
        rows, y, probs = p["test_rows"], p["y_true"], p["probs"]
        pred, conf = probs.argmax(1), probs.max(1)

        correct_items = []
        for c in range(config.NUM_CLASSES):
            ok = np.flatnonzero((y == c) & (pred == c))
            if len(ok) == 0:
                continue
            r = rows[rng.choice(ok)]
            img = np.asarray(images[r])
            cam, _, pr = gradcam.grad_cam(model, img.astype(np.float32), c)
            correct_items.append((img, cam, f"{NAMES[c]} ({pr:.2f})"))
        show_grid(correct_items, f"{name}: Grad-CAM on correctly classified test images",
                  config.FIGURES_DIR / f"gradcam_{name}_correct.png")

        wrong = np.flatnonzero(pred != y)
        wrong = wrong[np.argsort(-conf[wrong])][:24]
        err_items = []
        for i in wrong:
            img = np.asarray(images[rows[i]])
            cam, _, pr = gradcam.grad_cam(model, img.astype(np.float32), int(pred[i]))
            err_items.append((img, cam, f"true {NAMES[y[i]]} -> {NAMES[pred[i]]} ({pr:.2f})"))
        if err_items:
            show_grid(err_items, f"{name}: most confident mistakes (CAM for the predicted class)",
                      config.FIGURES_DIR / f"gradcam_{name}_errors.png", ncols=6)

        # quantitative check: does the model look at the hand?
        cand = [i for i in range(len(rows)) if detected[rows[i]] and y[i] != config.NOTHING_INDEX]
        cand = rng.choice(cand, size=min(args.focus_samples, len(cand)), replace=False) if cand else []
        foc, area = [], []
        for i in cand:
            r = rows[i]
            img = np.asarray(images[r]).astype(np.float32)
            cam, _, _ = gradcam.grad_cam(model, img, int(pred[i]))
            foc.append(gradcam.hand_focus(cam, lms[r]))
            xs, ys = lms[r][:, 0], lms[r][:, 1]
            area.append(min((xs.max() - xs.min()) * 1.3, 1) * min((ys.max() - ys.min()) * 1.3, 1))
        if foc:
            focus_rows.append({"model": name, "n": len(foc), "mean_cam_mass_in_hand_box": np.nanmean(foc),
                               "mean_box_area_fraction": np.mean(area)})
            print(focus_rows[-1])

    if focus_rows:
        pd.DataFrame(focus_rows).to_csv(config.RESULTS_DIR / "gradcam_hand_focus.csv", index=False)


if __name__ == "__main__":
    main()
