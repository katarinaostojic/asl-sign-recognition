"""Step 5: graph neural network over the hand skeleton (21 joints, 20 bones + palm links)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aslrec import models, training


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    training.add_common_args(ap, epochs=80, batch=256, lr=1e-3, patience=10)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 32])
    ap.add_argument("--dropout", type=float, default=0.3)
    args = ap.parse_args()
    name = args.name or "gnn"

    def build(norm):
        return models.build_gnn(norm, tuple(args.hidden), args.dropout, name=name)

    training.train_landmark_net(name, build, "graph", args)


if __name__ == "__main__":
    main()
