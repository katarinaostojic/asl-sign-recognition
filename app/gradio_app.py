"""Web demo (Gradio): classify a photo or spell words live from the webcam.

Run locally:      python app/gradio_app.py
Hugging Face:     see space/README.md (the same file is the Space's app.py)

Models are loaded from models/ (TFLite files are the lightest option for a hosted demo).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import gradio as gr
import numpy as np

from aslrec import config, inference
from aslrec.landmarks import HandDetector, draw_landmarks
from aslrec.wordbuilder import WordBuilder

NAMES = inference.available_model_names()
DEFAULT = next((n for n in ("mlp_tflite_float32", "mlp", "cnn_tflite_float32", "cnn") if n in NAMES),
               NAMES[0] if NAMES else None)

_photo_det = _cam_det = None
_pipes: dict = {}


def detectors():
    global _photo_det, _cam_det
    if _photo_det is None:
        _photo_det = HandDetector(pad=0.5)  # close-up photos
        _cam_det = HandDetector(pad=0)      # webcam frames
    return _photo_det, _cam_det


def pipeline(name):
    if name not in _pipes:
        _pipes[name] = inference.build_pipeline(name, detector=detectors()[0], crop_hand=True)
    return _pipes[name]


def top3(probs):
    idx = np.argsort(-probs)[:3]
    return {config.CLASS_NAMES[i].upper(): float(probs[i]) for i in idx}


def predict_photo(img, model_name):
    if img is None:
        return None, {}
    photo_det, _ = detectors()
    lm = photo_det.detect(img)
    probs = pipeline(model_name).predict_with_landmarks(img, lm)
    shown = img if lm is None else draw_landmarks(img, lm)
    return shown, top3(probs)


def stream_frame(frame, model_name, words):
    if frame is None:
        return None, "", words
    words = words or WordBuilder()
    _, cam_det = detectors()
    lm = cam_det.detect(frame)
    probs = pipeline(model_name).predict_with_landmarks(frame, lm)
    k = int(probs.argmax())
    state = words.update(config.CLASS_NAMES[k], float(probs[k]))
    shown = frame if lm is None else draw_landmarks(frame, lm)
    cv2.putText(shown, f"{config.CLASS_NAMES[k].upper()} {probs[k]:.0%}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return shown, state["text"], words


def clear():
    return "", WordBuilder()


with gr.Blocks(title="ASL fingerspelling recognition") as demo:
    gr.Markdown(
        "# ASL fingerspelling recognition\n"
        "MediaPipe hand landmarks → KNN / MLP / graph network, or a CNN on the hand crop. "
        "Static letters only (J and Z need motion)."
    )
    model = gr.Dropdown(NAMES, value=DEFAULT, label="Model")
    with gr.Tab("Photo"):
        with gr.Row():
            inp = gr.Image(type="numpy", sources=["upload", "clipboard"], label="Photo of a hand sign")
            out_img = gr.Image(label="Detected hand")
            out_lbl = gr.Label(num_top_classes=3, label="Prediction")
        inp.change(predict_photo, [inp, model], [out_img, out_lbl])
    with gr.Tab("Webcam: spell words"):
        gr.Markdown("Hold a sign steady to add the letter. SPACE and DELETE signs add a space / backspace. "
                    "Lower your hand between two identical letters.")
        words_state = gr.State(None)
        with gr.Row():
            cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Camera")
            live = gr.Image(label="Live view")
        text = gr.Textbox(label="Text", interactive=False, lines=2)
        cam.stream(stream_frame, [cam, model, words_state], [live, text, words_state], stream_every=0.15)
        gr.Button("Clear text").click(clear, None, [text, words_state])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
