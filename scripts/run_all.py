"""Run the whole pipeline step by step.

    python scripts/run_all.py --quick            # ~minutes: tiny subset, checks that everything runs
    python scripts/run_all.py                    # full experiment (hours on CPU, much faster on a GPU)
    python scripts/run_all.py --from 6           # continue from step 6 (e.g. after a crash)
    python scripts/run_all.py --only 3 4 9       # run selected steps

Steps: 1 prepare data | 2 landmarks | 3 KNN | 4 MLP | 5 GNN | 6 CNN | 7 MobileNetV2 | 8 CNN study
       9 evaluate | 10 Grad-CAM | 11 latency | 12 generalization | 13 TFLite export
(TFLite export runs before the latency benchmark so that the quantised models are included.)
"""
import argparse
import subprocess
import sys
from pathlib import Path

S = Path(__file__).resolve().parent


def steps(quick: bool):
    q = ["--quick"] if quick else []
    return {
        1: [["01_prepare_data.py"] + (["--per-class", "300", "--block-size", "10"] if quick else [])],
        2: [["02_extract_landmarks.py"]],
        3: [["03_train_knn.py"] + q, ["03_train_knn.py", "--features", "raw"] + q],
        4: [["04_train_mlp.py"] + q, ["04_train_mlp.py", "--features", "raw"] + q],
        5: [["05_train_gnn.py"] + q],
        6: [["06_train_cnn.py"] + q],
        7: [["07_train_mobilenet.py"] + q],
        8: [["08_cnn_hyperparameter_study.py"] + q],
        13: [["13_export_tflite.py", "--n-test", "300" if quick else "2000"]],
        9: [["09_evaluate.py"]],
        10: [["10_gradcam.py"] + (["--focus-samples", "50"] if quick else [])],
        11: [["11_benchmark_latency.py"] + (["--n", "30"] if quick else [])],
        12: [["12_generalization.py"] + (["--n", "120"] if quick else [])],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--only", type=int, nargs="*")
    args = ap.parse_args()

    plan = steps(args.quick)
    order = [1, 2, 3, 4, 5, 6, 7, 8, 13, 9, 10, 11, 12]
    rank = {n: (8.5 if n == 13 else n) for n in order}
    for n in order:
        if args.only and n not in args.only:
            continue
        if not args.only and rank[n] < args.start:
            continue
        for cmd in plan[n]:
            print(f"\n===== step {n}: {' '.join(cmd)} =====", flush=True)
            r = subprocess.run([sys.executable, str(S / cmd[0])] + cmd[1:])
            if r.returncode != 0:
                sys.exit(f"Step {n} failed (exit code {r.returncode}). Fix it, then resume with --from {n}.")
    print("\nAll done. See results/REPORT.md and results/figures/.")


if __name__ == "__main__":
    main()
