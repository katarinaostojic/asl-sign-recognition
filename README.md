# ASL Fingerspelling Recognition: hand landmarks vs. images

Recognising the American Sign Language alphabet from a single photo or a live webcam stream,
and answering a concrete question: **are MediaPipe hand landmarks enough, or does a model that
sees the whole image do better?** Five models are compared on the same data, with a careful
evaluation (no frame leakage, honest handling of failed hand detections, calibration,
robustness to a change of conditions, speed and model size) and a runnable demo.

| | Model | Input |
|---|---|---|
| Baseline | **KNN** | 94 engineered features from 21 MediaPipe landmarks |
| Architecture 1 | **MLP** | the same features |
| Extension | **Graph neural network** over the hand skeleton | 21 joints × 6 node features |
| Architecture 2 | **CNN** trained from scratch (+ Grad-CAM) | 96×96 RGB image |
| Extension | **MobileNetV2** transfer learning (fine-tuned) | 96×96 RGB image |

Everything is in English, driven by scripts (`scripts/01…13`), unit-tested where it can be
(`tests/`), and every run is logged under `results/runs/`.

## Research questions

1. Are MediaPipe landmarks sufficient for classifying ASL letters? → KNN / MLP / GNN accuracy, per-class F1, `results/REPORT.md`.
2. Does a model that sees the complete image beat one that only sees hand geometry? → CNN / MobileNetV2 vs. landmark models, compared on the **common subset** (identical images for every model).
3. How far apart are a classical algorithm and neural networks? → KNN vs. MLP / GNN on identical features, with significance tests (McNemar) and block-bootstrap confidence intervals.

Beyond the three questions: *does a 99 % test accuracy survive a change of conditions?* (`scripts/12_generalization.py`),
*are the models' confidences trustworthy?* (reliability diagrams, accuracy-vs-coverage), *what do the CNNs look at?*
(Grad-CAM + a quantitative hand-focus score) and *what does deployment cost?* (TFLite, quantisation, latency).

## Results

> Run the pipeline (see **Quick start**), then paste the tables from `results/REPORT.md` here.
> All figures are written to `results/figures/`:
> `confusion_matrices.png`, `per_class_f1.png`, `accuracy_comparison.png`, `confidence_analysis.png`,
> `gradcam_*_correct.png`, `gradcam_*_errors.png`, `robustness.png`, `tflite_comparison.png`,
> `cnn_hyperparameter_study.png`, `detection_rate.png`.

## Dataset and evaluation protocol

**Data.** [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet): 87,000 training images (200×200), 29 classes
(A–Z, `space`, `del`, `nothing`), 3,000 per class. Check the licence on the Kaggle page before redistributing anything derived from it.
The 29-image `asl_alphabet_test` folder is far too small for evaluation and is not used.

**Why the split is blockwise.** The images are consecutive frames of short videos, so neighbouring images are
near-duplicates. A random split puts almost identical frames into train and test and inflates the accuracy.
`scripts/01_prepare_data.py` therefore keeps blocks of 100 consecutive frames together and assigns whole
blocks to train / validation / test (≈70 / 13 / 17 % per class). Use `--split random` to reproduce the leaky
protocol and see the difference. Confidence intervals resample whole blocks for the same reason.

**Failed hand detections.** MediaPipe does not find a hand in every image. Those images are *not* dropped silently:

* the landmark models are trained and tuned on detected images only;
* the **end-to-end** view scores every test image and treats a missed detection as the answer `nothing`
  (correct for the class `nothing`, a mistake for all others), which is what a real system would do;
* the **common** view restricts all models, including the CNNs, to images where a hand was found, which is the fair
  way to compare landmarks with full images (identical inputs).

The detection rate per class is stored in `results/detection_rate.csv`. Because the photos are tight close-ups,
detection runs on a padded and upscaled copy of the image (`landmarks.HandDetector`).

**Features.** Landmarks are translated to the wrist and scaled by the wrist→middle-finger-base distance (removes hand
position and size, keeps orientation). *Raw* = 60 coordinates; *engineered* = raw + 15 joint-angle cosines + 4 finger-spread
angles + 10 fingertip distances + 5 tip-to-wrist distances (94 values). `scripts/03` and `04` accept `--features raw` for an ablation.

**Graph network.** Joints are nodes, bones plus palm links are edges (symmetric-normalised adjacency), node features are the
normalised coordinates plus the bone vector to the parent joint; three graph-convolution layers → dense head. Implemented in
plain Keras (`models.GraphConv`), no extra dependency.

**Images.** 96×96 RGB, cached once as a `uint8` array; augmentation (rotation, translation, zoom, brightness, contrast) is applied on
the fly in `tf.data` for training only. The CNN has `--blocks`, `--kernel`, `--activation`, `--dropout` options, and
`scripts/08_cnn_hyperparameter_study.py` varies each of them on the validation split.
MobileNetV2 is trained in two stages (new head, then fine-tuning the last layers).

**Metrics.** Accuracy, macro precision / recall / F1, confusion matrices and the most frequent confusions, expected calibration error,
reliability diagrams, accuracy-vs-coverage (reject the least confident predictions), pairwise McNemar tests, bootstrap CIs,
parameters, file size, training time and prediction time per image (MediaPipe included).

## Quick start

Requirements: Python 3.10–3.12. TensorFlow ≥ 2.11 has **no GPU support on native Windows**; for GPU training use WSL2 or Google Colab
(`notebooks/run_on_colab.ipynb`, one click per cell). The landmark models and the demo run fine on a laptop CPU.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

# 1. place the Kaggle dataset in ./dataset  (the folder that contains asl_alphabet_train/)
python scripts/run_all.py --quick   # smoke test on a tiny subset (minutes); results are meaningless
python scripts/run_all.py           # full experiment
```

or step by step:

| Step | Script | What it does |
|---|---|---|
| 1 | `01_prepare_data.py` | index, blockwise split, image cache, example grid, class balance |
| 2 | `02_extract_landmarks.py` | MediaPipe on all images, detection rate per class |
| 3 | `03_train_knn.py` | KNN baseline, k and weighting chosen on validation |
| 4 | `04_train_mlp.py` | MLP (`--features raw` or `engineered`) |
| 5 | `05_train_gnn.py` | graph network over the hand skeleton |
| 6 | `06_train_cnn.py` | CNN from scratch |
| 7 | `07_train_mobilenet.py` | MobileNetV2 transfer learning + fine-tuning |
| 8 | `08_cnn_hyperparameter_study.py` | depth / kernel / activation / dropout study |
| 13 | `13_export_tflite.py` | TFLite float32 / dynamic / fp16 / int8 + comparison |
| 9 | `09_evaluate.py` | all metrics, figures, `results/REPORT.md` |
| 10 | `10_gradcam.py` | Grad-CAM grids and hand-focus score |
| 11 | `11_benchmark_latency.py` | prediction time per model |
| 12 | `12_generalization.py` | perturbation stress test and your own photos |

Every training script has `--help`, `--seed`, `--epochs`, `--name`, and `--quick`. Run the whole thing with different seeds and
different `--name`s to estimate run-to-run variance. Set `ASL_MLFLOW=1` (with `pip install mlflow`) to also log runs to MLflow.

## Testing on your own photos

```bash
python tools/capture_photos.py      # webcam: press a-z (letter), 1 space, 2 del, 3 nothing
python scripts/12_generalization.py # accuracy on data/custom/ for every model
```

Vary background, lighting and distance. The gap between the test accuracy and the accuracy on your photos is the most honest number in this project.
Image models are evaluated on the whole photo and on a hand-centred crop (the training images are close-ups).

## Demo

```bash
python app/webcam_demo.py --model mlp          # OpenCV window, live spelling (Esc quits, c clears, m switches model)
pip install -r requirements-demo.txt
python app/gradio_app.py                       # browser UI: photo upload + webcam word builder
docker build -t asl-demo . && docker run -p 7860:7860 asl-demo
```

Predictions are smoothed with a majority vote over recent frames (`wordbuilder.WordBuilder`): holding a sign adds the letter,
`space` and `del` act as space and backspace, `nothing` re-arms the builder so repeated letters are possible.
To publish a public demo see `space/README.md` (Hugging Face Spaces).

## Project layout

```
src/aslrec/     config, data, splits, features, landmarks, models, training, metrics,
                gradcam, inference, robustness, wordbuilder, tracking
scripts/        numbered pipeline steps + run_all.py
app/            webcam_demo.py (OpenCV), gradio_app.py (web)
tools/          capture_photos.py
tests/          unit tests (features, splits, metrics, word builder, perturbations)
notebooks/      run_on_colab.ipynb
space/          Hugging Face Spaces instructions
results/        summary.csv, REPORT.md, figures/, per-run logs (created by the scripts)
```

`python -m pytest` runs the unit tests (they need only NumPy / pandas / scikit-learn / OpenCV); CI runs them on every push.

## Limitations

* Only the **static** letters are recognised; J and Z involve motion, so they are only approximated by a single frame.
* The dataset has very little variety in people, backgrounds and lighting; results on it do not transfer automatically to the real world, which is what the robustness and own-photo tests measure.
* Hand detection is a prerequisite for the landmark models; their end-to-end accuracy includes this failure mode by design.
* Latencies depend on the machine; report them together with the hardware (`results/latency.csv` records it).
