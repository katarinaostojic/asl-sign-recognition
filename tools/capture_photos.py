"""Record your own test photos with the webcam (for scripts/12_generalization.py).

Usage:  python tools/capture_photos.py
  - Show a sign, then press the matching key: a-z saves a letter, 1 = space, 2 = del, 3 = nothing.
  - Vary background, lighting and distance between photos; aim for 5+ photos per letter.
  - Esc quits. Photos go to data/custom/<CLASS>/<CLASS>_<n>.jpg (kept out of git).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2

from aslrec import config

SPECIAL = {ord("1"): "space", ord("2"): "del", ord("3"): "nothing"}


def main():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise SystemExit("Could not open the webcam.")
    counts = {}
    last = ""
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        shown = frame.copy()
        cv2.putText(shown, f"a-z letter | 1 space | 2 del | 3 nothing | Esc quit   saved: {last}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow("capture", shown)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        label = SPECIAL.get(k) or (chr(k).upper() if 97 <= k <= 122 else None)
        if label:
            d = config.CUSTOM_DIR / label
            d.mkdir(parents=True, exist_ok=True)
            counts[label] = counts.get(label, len(list(d.glob("*.jpg")))) + 1
            path = d / f"{label}_{counts[label]:03d}.jpg"
            cv2.imwrite(str(path), frame)
            last = path.name
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
