---
title: ASL Fingerspelling Recognition
emoji: 🤟
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# Deploying the demo on Hugging Face Spaces

1. Create a new Space (SDK: Gradio, hardware: CPU basic) at <https://huggingface.co/new-space>.
2. Copy these into the Space repository:

   ```
   app.py                 <- copy of app/gradio_app.py
   requirements.txt       <- copy of requirements-demo.txt
   src/aslrec/...         <- the src/aslrec folder
   models/tflite/*.tflite <- exported models (scripts/13_export_tflite.py), e.g. mlp_tflite_float32.tflite
   models/*.joblib        <- optional (KNN)
   ```
3. Replace this file by the Space's `README.md` (the YAML header above is required by Spaces).
4. Push. The hand-landmark model is downloaded on first start from Google's model storage.

Only TFLite and scikit-learn models are loaded in the Space, so TensorFlow is not needed
(`ai-edge-litert` runs the `.tflite` files). Keep Space models small: the MLP / graph models
are a few hundred KB; the MobileNetV2 variants are a few MB.
