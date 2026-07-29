# Eye Vision Pipeline

Two small computer-vision proof-of-concepts exploring eye tracking
and retinal image analysis: a real-time gaze estimator (webcam +
MediaPipe) and a retinal fundus image classifier (PyTorch transfer
learning) deployed as a live Streamlit web app.

**Live demo:** https://eye-vision-pipeline-8upzaawhrkdrwrly4ysysv.streamlit.app/

## Modules

### `gaze_tracker/`
Real-time gaze direction estimation from a webcam feed.
- **Pipeline:** capture frame -> detect iris/eye-corner landmarks
  (MediaPipe Face Mesh, `refine_landmarks=True`) -> compute horizontal
  gaze ratio per eye -> classify as LEFT / RIGHT / CENTER -> overlay
  on video.
- **Run it:** `python -m gaze_tracker.main` (press `q` to quit, `l`
  to log a reading to `gaze_log.csv`)
- **Known limitations:** no per-user calibration, sensitive to head
  pose and lighting. A production system would calibrate per user
  and/or use a trained regression model instead of a fixed geometric
  threshold.

### `retinal_classifier/`
Binary classifier (No DR vs. Moderate DR) for retinal fundus images,
trained locally with PyTorch using transfer learning on a pretrained
ResNet18. See `retinal_classifier/README.md` for the full pipeline
walkthrough and dataset source.
- **Pipeline:** collect (APTOS 2019 sample, ~40 images) -> preprocess
  (OpenCV: crop, resize, CLAHE contrast enhancement) -> train
  (frozen ResNet18 backbone, retrained final layer) -> evaluate
  (80/20 train/val split) -> predict.
- **Train it:** `python -m retinal_classifier.train`
- **Predict locally:** `python -m retinal_classifier.predict_local --image <path>`
- Also includes `predict.py`, a ready-to-use client for an Azure
  Custom Vision prediction endpoint — written but not deployed (see
  Next steps).

### `app.py`
Streamlit web app wrapping the retinal classifier: upload a fundus
image, get live predictions in-browser. Deployed for free on
Streamlit Community Cloud, connected directly to this repo.

## Setup

**Web app / retinal classifier only (lighter install):**
