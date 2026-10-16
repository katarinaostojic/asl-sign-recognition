"""Step 3: baseline - K-Nearest Neighbours on MediaPipe landmark features.

k and the vote weighting are chosen on the validation split; the test split is only
used once at the end. The model is saved as a scikit-learn pipeline (scaler + KNN).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aslrec import config, data, tracking

KS = [1, 3, 5, 7, 11, 15, 21]


def vote(labels_nb, dists, k, weights, n_classes):
    n = len(labels_nb)
    lab = labels_nb[:, :k]
    w = np.ones_like(dists[:, :k]) if weights == "uniform" else 1.0 / (dists[:, :k] + 1e-6)
    votes = np.zeros((n, n_classes))
    np.add.at(votes, (np.repeat(np.arange(n), k), lab.ravel()), w.ravel())
    return votes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=["raw", "engineered"], default="engineered")
    ap.add_argument("--name", default=None, help="default: knn (engineered) or knn_raw")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    name = args.name or ("knn" if args.features == "engineered" else "knn_raw")

    df, X, detected = data.load_landmark_arrays(args.features)
    y = df["label"].to_numpy()
    tr, va, te = (data.split_rows(df, s) for s in ("train", "val", "test"))
    if args.quick:
        tr, va, te = (data.subsample_rows(df, r, n) for r, n in ((tr, 300), (va, 60), (te, 60)))
    tr, va = tr[detected[tr]], va[detected[va]]
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}")

    scaler = StandardScaler().fit(X[tr])
    Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
    kmax = min(max(KS), len(tr))
    nn = KNeighborsClassifier(n_neighbors=kmax, n_jobs=-1).fit(Xtr, y[tr])
    dist, idx = nn.kneighbors(Xva)
    labels_nb = y[tr][idx]

    rows = []
    for weights in ("uniform", "distance"):
        for k in KS:
            if k > kmax:
                continue
            pred = vote(labels_nb, dist, k, weights, config.NUM_CLASSES).argmax(1)
            rows.append({"k": k, "weights": weights, "val_accuracy": float((pred == y[va]).mean())})
    grid = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
    print(grid.to_string(index=False))
    best = grid.iloc[0]
    config.RUNS_DIR.joinpath(name).mkdir(parents=True, exist_ok=True)
    grid.to_csv(config.RUNS_DIR / name / "grid.csv", index=False)

    t0 = time.time()
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=int(best["k"]), weights=best["weights"], n_jobs=-1),
    ).fit(X[tr], y[tr])
    t_fit = time.time() - t0

    det = detected[te]
    probs = np.zeros((len(te), config.NUM_CLASSES), dtype=np.float32)
    if det.any():
        p = model.predict_proba(X[te][det])
        probs[np.ix_(det, model.classes_)] = p
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODELS_DIR / f"{name}.joblib")
    tracking.save_predictions(name, te, y[te], probs, det)
    acc = float((probs[det].argmax(1) == y[te][det]).mean()) if det.any() else float("nan")
    tracking.save_run(name, {"model": name, "features": args.features, "k": int(best["k"]),
                             "weights": best["weights"], "quick": args.quick},
                      {"test_accuracy_on_detected": acc, "best_val_accuracy": float(best["val_accuracy"]),
                       "test_detected_fraction": float(det.mean()), "train_seconds": t_fit,
                       "parameters": int(len(tr) * X.shape[1]),  # stored training set
                       "model_file_mb": (config.MODELS_DIR / f"{name}.joblib").stat().st_size / 1e6})
    print(f"\n[{name}] k={best['k']} weights={best['weights']}  "
          f"val={best['val_accuracy']:.4f}  test(on detected)={acc:.4f}")


if __name__ == "__main__":
    main()
