"""Real-time ASL fingerspelling demo with the webcam (OpenCV window).

    python app/webcam_demo.py --model mlp
    python app/webcam_demo.py --model cnn            # image model on a hand-centred crop
    python app/webcam_demo.py --model mlp_tflite_int8 # a TFLite file from models/tflite/

Keys:  Esc quit | c clear text | m switch to the next available model
Hold a sign steady until the progress bar fills to add the letter; show SPACE / DELETE signs
for a space / backspace. Lower your hand between two identical letters.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from aslrec import config, inference
from aslrec.landmarks import HandDetector, draw_landmarks
from aslrec.wordbuilder import WordBuilder


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    names = inference.available_model_names()
    if not names:
        raise SystemExit("No trained models in models/. Train some first (scripts/03..07).")
    if args.model not in names:
        print(f"Model '{args.model}' not found, using '{names[0]}'. Available: {names}")
        args.model = names[0]

    detector = HandDetector(pad=0)  # webcam frames: the hand is already small, no padding needed
    pipe = inference.build_pipeline(args.model, detector=detector, crop_hand=True)
    words = WordBuilder()
    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        raise SystemExit("Could not open the webcam.")
    fps, last = 0.0, time.perf_counter()

    while True:
        ok, bgr = cam.read()
        if not ok:
            break
        bgr = cv2.flip(bgr, 1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lm = detector.detect(rgb)
        probs = pipe.predict_with_landmarks(rgb, lm)
        k = int(probs.argmax())
        label, conf = config.CLASS_NAMES[k], float(probs[k])
        state = words.update(label, conf)

        view = bgr if lm is None else cv2.cvtColor(draw_landmarks(rgb, lm), cv2.COLOR_RGB2BGR)
        h, w = view.shape[:2]
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        cv2.rectangle(view, (0, 0), (w, 70), (30, 30, 30), -1)
        cv2.putText(view, f"{label.upper()}  {conf:.0%}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        cv2.rectangle(view, (10, 52), (10 + int(200 * state["progress"]), 62), (0, 200, 255), -1)
        cv2.putText(view, f"{args.model}  {fps:.0f} fps", (w - 260, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        cv2.rectangle(view, (0, h - 50), (w, h), (30, 30, 30), -1)
        cv2.putText(view, state["text"][-40:] + "_", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("ASL fingerspelling", view)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("c"):
            words.reset()
        if key == ord("m"):
            args.model = names[(names.index(args.model) + 1) % len(names)]
            pipe = inference.build_pipeline(args.model, detector=detector, crop_hand=True)
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
