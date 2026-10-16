"""Step 7: transfer learning - MobileNetV2 (ImageNet) with a new head, then fine-tuning.

Stage 1 trains only the new head (backbone frozen). Stage 2 unfreezes the last layers of
the backbone and continues with a small learning rate.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aslrec import config, models, training


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    training.add_common_args(ap, epochs=8, batch=128, lr=1e-3, patience=3)
    ap.add_argument("--finetune-epochs", type=int, default=12)
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--unfreeze", type=int, default=40, help="number of backbone layers to fine-tune")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (ablation)")
    args = ap.parse_args()
    name = args.name or "mobilenet"

    model = models.build_mobilenet(config.IMG_SIZE, args.dropout,
                                   weights=None if args.no_pretrained else "imagenet", name=name)
    ft = args.finetune_epochs if not args.quick else 1
    training.train_image_net(
        name, model, args,
        finetune_fn=lambda m: models.unfreeze_backbone(m, args.unfreeze),
        finetune_epochs=ft, finetune_lr=args.finetune_lr,
    )


if __name__ == "__main__":
    main()
