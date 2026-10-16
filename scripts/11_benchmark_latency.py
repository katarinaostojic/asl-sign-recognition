"""Step 11: prediction time per image for every model (end-to-end and per stage).

Measures, on `--n` test images with batch size 1 on THIS machine (single thread where
possible): the full pipeline (MediaPipe + features + model, or resize + CNN), and the MediaPipe
stage alone. Warm-up runs are excluded. Output: results/latency.csv

Latency depends strongly on the hardware; always report it together with the CPU/GPU used.
"""
import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from aslrec import config, data, inference
from aslrec.imaging import read_rgb
from aslrec.landmarks import HandDetector


def timeit(fn, imgs, warmup=10):
    for img in imgs[:warmup]:
        fn(img)
    t = []
    for img in imgs:
        s = time.perf_counter()
        fn(img)
        t.append((time.perf_counter() - s) * 1000)
    return np.array(t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--models", nargs="*", help="default: every model found in models/")
    args = ap.parse_args()

    df = data.load_index()
    te = data.split_rows(df, "test")
    rng = np.random.default_rng(0)
    pick = rng.choice(te, size=min(args.n, len(te)), replace=False)
    imgs = [read_rgb(df["path"].iloc[i]) for i in pick]

    detector = HandDetector()
    names = args.models
    if not names:
        names = [n for n in inference.MODEL_KINDS if any(
            (config.MODELS_DIR / f"{n}{s}").exists() for s in (".joblib", ".keras"))]
        names += sorted(p.stem for p in (config.MODELS_DIR / "tflite").glob("*.tflite")) \
            if (config.MODELS_DIR / "tflite").exists() else []

    rows = []
    det_t = timeit(detector.detect, imgs)
    rows.append({"model": "(MediaPipe detection only)", "mean_ms": det_t.mean(),
                 "median_ms": np.median(det_t), "p95_ms": np.percentile(det_t, 95)})
    for name in names:
        pipe = inference.build_pipeline(name, detector=detector)
        t = timeit(lambda im: pipe.predict_rgb([im]), imgs)
        rows.append({"model": name, "mean_ms": t.mean(), "median_ms": np.median(t),
                     "p95_ms": np.percentile(t, 95)})
        print(rows[-1])
    out = pd.DataFrame(rows)
    out["machine"] = f"{platform.processor() or platform.machine()} / {platform.system()}"
    out.to_csv(config.RESULTS_DIR / "latency.csv", index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
