"""Step 2: run MediaPipe Hands on every image and store the 21 landmarks.

Images where MediaPipe finds no hand are flagged (`detected == False`). They are NOT
dropped silently: the landmark models are trained on detected images only, but the
final evaluation counts a missed detection as a mistake (except for the class
"nothing", where "no hand found" is the correct answer). See README, "Handling
failed detections".

Outputs
  data/cache/landmarks.npz                 landmarks (N, 21, 3) + detected (N,)
  results/detection_rate.csv               detection rate per class
  results/figures/detection_rate.png
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

from aslrec import config, data, landmarks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    ap.add_argument("--limit", type=int, help="only process the first N rows (smoke test)")
    args = ap.parse_args()

    df = data.load_index()
    paths = df["path"].tolist()
    if args.limit:
        paths = paths[: args.limit]
    lms, ok = landmarks.extract_from_paths(paths, args.workers)
    if args.limit:  # pad so that arrays stay aligned with splits.csv
        full_l = np.zeros((len(df), 21, 3), np.float32)
        full_o = np.zeros(len(df), bool)
        full_l[: len(paths)], full_o[: len(paths)] = lms, ok
        lms, ok = full_l, full_o

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.LANDMARKS_NPZ, landmarks=lms, detected=ok)

    df = df.assign(detected=ok)
    rate = df.groupby("class_name")["detected"].mean().reindex(config.CLASS_NAMES)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rate.rename("detection_rate").to_csv(config.RESULTS_DIR / "detection_rate.csv")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ax = rate.plot(kind="bar", figsize=(12, 3.5))
    ax.set_ylabel("hand detected")
    ax.set_ylim(0, 1.02)
    ax.set_title("MediaPipe detection rate per class")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "detection_rate.png", dpi=150)
    plt.close()
    print(f"Overall detection rate: {ok.mean():.3f}")
    print(rate.round(3).to_string())


if __name__ == "__main__":
    main()
