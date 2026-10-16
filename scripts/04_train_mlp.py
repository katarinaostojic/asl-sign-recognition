"""Step 4: Architecture 1 - MLP on MediaPipe landmark features."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aslrec import features, models, training


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    training.add_common_args(ap, epochs=80, batch=256, lr=1e-3, patience=10)
    ap.add_argument("--features", choices=["raw", "engineered"], default="engineered")
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--activation", default="relu")
    args = ap.parse_args()
    name = args.name or ("mlp" if args.features == "engineered" else "mlp_raw")

    dim = 60 if args.features == "raw" else 94

    def build(norm):
        return models.build_mlp(dim, norm, tuple(args.hidden), args.dropout, args.activation, name=name)

    training.train_landmark_net(name, build, args.features, args)


if __name__ == "__main__":
    main()
