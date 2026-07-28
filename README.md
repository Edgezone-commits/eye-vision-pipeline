# Eye Vision Pipeline

Two small computer-vision proof-of-concepts exploring eye tracking
and retinal image analysis: a real-time gaze estimator (webcam +
MediaPipe) and a retinal image classifier deployed via Azure Custom
Vision.

## Modules

### `gaze_tracker/`
Real-time gaze direction estimation from a webcam feed.
- **Pipeline:** capture frame -> detect iris/eye-corner landmarks
  (MediaPipe Face Mesh) -> compute horizontal gaze ratio -> classify
  as LEFT / RIGHT / CENTER -> overlay on video.
- **Run it:** `python -m gaze_tracker.main`
- **Known limitations:** no per-user calibration, sensitive to head
  pose and lighting. A production system would calibrate per user
  and/or use a trained regression model instead of a fixed geometric
  threshold.

### `retinal_classifier/`
Retinal fundus image classifier, trained and deployed through Azure
Custom Vision, tested via a small Python client. See
`retinal_classifier/README.md` for the full walkthrough and dataset
source.

## Setup

```
git clone <this repo>
cd eye-vision-pipeline
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Why these two modules

Built while exploring the tools listed for an eye-tracking /
retinal-scanning computer vision role: OpenCV, MediaPipe, a deep
learning framework, and Azure. I hadn't worked with retinal imaging
or Azure Cognitive Services before this, so the retinal classifier
module doubled as a first hands-on pass at both.

## Next steps

- Add per-user calibration to the gaze tracker.
- Replace Custom Vision with a custom-trained PyTorch model on a
  larger retinal dataset, deployed via Azure ML.
- Add automated evaluation scripts (accuracy/precision/recall logging)
  instead of manual review in the Custom Vision portal.
