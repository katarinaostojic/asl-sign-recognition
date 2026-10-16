"""Central configuration: paths, class names and global constants."""
from __future__ import annotations

import os
import string
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PRED_DIR = RESULTS_DIR / "predictions"
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = RESULTS_DIR / "figures"
CUSTOM_DIR = DATA_DIR / "custom"  # your own photos: data/custom/<CLASS>/*.jpg

SPLITS_CSV = DATA_DIR / "splits.csv"
LANDMARKS_NPZ = CACHE_DIR / "landmarks.npz"

IMG_SIZE = 96          # input size for CNN / MobileNetV2 (cache is built at this size)
SEED = 42
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train / val / test
BLOCK_SIZE = 100       # consecutive frames kept together when splitting (see README)

# Folder names in the Kaggle dataset, in a fixed order.
CLASS_NAMES = list(string.ascii_uppercase) + ["del", "nothing", "space"]
NUM_CLASSES = len(CLASS_NAMES)
NOTHING_INDEX = CLASS_NAMES.index("nothing")
SPACE_INDEX = CLASS_NAMES.index("space")
DEL_INDEX = CLASS_NAMES.index("del")

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"


def find_train_dir(explicit: str | os.PathLike | None = None) -> Path:
    """Locate the folder that directly contains the class folders (A, B, ..., nothing, space)."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("ASL_DATASET_DIR"):
        candidates.append(Path(os.environ["ASL_DATASET_DIR"]))
    candidates += [ROOT / "dataset", ROOT, ROOT / "data" / "raw"]

    def is_train_dir(p: Path) -> bool:
        return (p / "A").is_dir() and (p / "nothing").is_dir() and (p / "space").is_dir()

    for base in candidates:
        for sub in (
            base,
            base / "asl_alphabet_train",
            base / "asl_alphabet_train" / "asl_alphabet_train",
        ):
            if sub.is_dir() and is_train_dir(sub):
                return sub
    raise FileNotFoundError(
        "Could not find the ASL Alphabet training folder. Pass --dataset <path> or set "
        "ASL_DATASET_DIR. Expected a folder containing the class folders A, B, ..., nothing, space."
    )
