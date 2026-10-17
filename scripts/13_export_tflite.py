"""Step 13: export models to TensorFlow Lite and compare size, speed and accuracy.

For every model it creates four variants in models/tflite/:
  <name>_tflite_float32.tflite   plain conversion
  <name>_tflite_dynamic.tflite   dynamic-range quantisation (int8 weights)
  <name>_tflite_fp16.tflite      float16 weights
  <name>_tflite_int8.tflite      full-integer quantisation of weights and activations
                                 (calibrated on validation samples; inputs/outputs stay float)
and compares them with the Keras model on test samples (accuracy, latency with 1 thread, file size).
Landmark models are scored on the test images where a hand was detected.

Output: results/tflite_comparison.csv and results/figures/tflite_comparison.png
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
import numpy as np
import pandas as pd
import tensorflow as tf

from aslrec import config, data, features, inference, models  # noqa: F401

VARIANTS = ["float32", "dynamic", "fp16", "int8"]


def inputs_for(name, df, rows):
    kind, feats = inference.describe(name)
    if kind == "image":
        return np.asarray(data.load_images()[np.sort(rows)]).astype(np.float32), np.sort(rows)
    z = np.load(config.LANDMARKS_NPZ)
    rows = np.sort(rows[z["detected"][rows]])
    return features.FEATURE_BUILDERS[feats](z["landmarks"][rows]), rows


def convert(model, variant, rep_x):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if variant != "float32":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if variant == "fp16":
        conv.target_spec.supported_types = [tf.float16]
    if variant == "int8":
        def rep():
            for s in rep_x:
                yield [s[None].astype(np.float32)]
        conv.representative_dataset = rep
    return conv.convert()


def latency_ms(pred, x, n=200):
    x = x[:n]
    pred.predict_proba(x[:5])
    t = []
    for s in x:
        a = time.perf_counter()
        pred.predict_proba(s[None])
        t.append((time.perf_counter() - a) * 1000)
    return float(np.mean(t))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["mlp", "gnn", "cnn", "mobilenet"])
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-calib", type=int, default=200)
    args = ap.parse_args()

    df = data.load_index()
    y_all = df["label"].to_numpy()
    out_dir = config.MODELS_DIR / "tflite"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    results = []

    for name in args.models:
        kpath = config.MODELS_DIR / f"{name}.keras"
        if not kpath.exists():
            print(f"skip {name}: {kpath} not found")
            continue
        model = keras.models.load_model(kpath, compile=False)
        pool = data.split_rows(df, "test")
        te = rng.choice(pool, size=min(args.n_test, len(pool)), replace=False)
        x_te, rows = inputs_for(name, df, te)
        y_te = y_all[rows]
        pool = data.split_rows(df, "val")
        va = rng.choice(pool, size=min(args.n_calib * 2, len(pool)), replace=False)
        x_cal, _ = inputs_for(name, df, va)
        x_cal = x_cal[: args.n_calib]

        kp = inference.KerasPredictor(kpath)
        results.append({"model": name, "variant": "keras", "accuracy": float((kp.predict_proba(x_te).argmax(1) == y_te).mean()),
                        "latency_ms": latency_ms(kp, x_te), "size_mb": kpath.stat().st_size / 1e6, "n": len(y_te)})
        print(results[-1])
        for v in VARIANTS:
            path = out_dir / f"{name}_tflite_{v}.tflite"
            try:
                path.write_bytes(convert(model, v, x_cal))
                tp = inference.TFLitePredictor(path)
                acc = float((tp.predict_proba(x_te).argmax(1) == y_te).mean())
                results.append({"model": name, "variant": v, "accuracy": acc, "latency_ms": latency_ms(tp, x_te),
                                "size_mb": path.stat().st_size / 1e6, "n": len(y_te)})
            except Exception as e:  # keep going: some ops may not convert / quantise
                print(f"  {name}/{v} failed: {type(e).__name__}: {str(e)[:200]}")
                path.unlink(missing_ok=True)  # do not leave an unloadable file in models/tflite
                continue
            print(results[-1])

    res = pd.DataFrame(results)
    res.to_csv(config.RESULTS_DIR / "tflite_comparison.csv", index=False)
    if len(res):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, col, title in zip(axes, ["accuracy", "latency_ms", "size_mb"],
                                  ["Accuracy", "Latency per sample (ms, 1 thread)", "File size (MB)"]):
            res.pivot(index="model", columns="variant", values=col).plot(kind="bar", ax=ax, rot=0)
            ax.set_title(title)
            if col == "accuracy":
                ax.set_ylim(max(0, res["accuracy"].min() - 0.05), 1.0)
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "tflite_comparison.png", dpi=150)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
