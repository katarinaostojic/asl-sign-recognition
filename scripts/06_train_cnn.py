"""Step 6: Architecture 2 - CNN trained from scratch on the full image (96x96)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aslrec import config, models, training


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    training.add_common_args(ap, epochs=40, batch=128, lr=1e-3, patience=6)
    ap.add_argument("--blocks", type=int, default=4, help="number of conv blocks (2 convs each)")
    ap.add_argument("--filters", type=int, default=32, help="filters in the first block")
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--activation", default="relu")
    ap.add_argument("--dropout", type=float, default=0.4)
    args = ap.parse_args()
    name = args.name or "cnn"

    model = models.build_cnn(config.IMG_SIZE, args.blocks, args.filters, args.kernel,
                             args.activation, args.dropout, name=name)
    training.train_image_net(name, model, args)


if __name__ == "__main__":
    main()
