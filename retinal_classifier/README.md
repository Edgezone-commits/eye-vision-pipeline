# Retinal Image Classifier (Azure Custom Vision)

A small proof-of-concept pipeline for classifying retinal fundus
images, using Azure Custom Vision as the training/deployment
platform. Chosen deliberately over training a PyTorch model from
scratch: with a small labeled dataset and limited time, Custom
Vision's transfer-learning backend gets a working, deployed
classifier in under an hour, and still requires understanding the
same CV pipeline stages (collect -> preprocess -> label -> train ->
evaluate -> deploy).

## Dataset

Sample images pulled from the [APTOS 2019 Blindness Detection]
(https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)
dataset on Kaggle (public, requires free Kaggle account). Only a
small subset (~15-20 images per class) was used here as a
proof-of-concept — not the full competition dataset.

Classes used: `No_DR` (no diabetic retinopathy) and `Moderate_DR`.

## Pipeline steps

1. **Collect** — download a small labeled subset from Kaggle.
2. **Preprocess** — run `preprocess.py` to resize, center-crop, and
   contrast-enhance each image before upload.
3. **Label / Train** — upload processed images to a Custom Vision
   project via the web portal, tag each by class, and train.
4. **Evaluate** — review precision/recall per tag in the Custom
   Vision portal's Performance tab.
5. **Deploy** — publish the trained iteration to a prediction
   endpoint.
6. **Test via code** — `predict.py` calls that endpoint from Python
   and prints the predicted class probabilities.

## How to reproduce (step-by-step)

1. Go to https://www.customvision.ai and sign in with a Microsoft
   account (Azure free tier is enough).
2. Create a new project: Project Type = **Classification**,
   Classification Type = **Multiclass**, Domain = **General**.
3. Upload preprocessed images, adding one tag per class as you go
   (`No_DR`, `Moderate_DR`).
4. Click **Train** (Quick Training).
5. Check the **Performance** tab for precision/recall per tag.
6. Click **Publish**, name the iteration, and note down:
   - the **Prediction URL** (goes in `AZURE_CV_ENDPOINT_URL`)
   - the **Prediction Key** (goes in `AZURE_CV_PREDICTION_KEY`)
7. Run `python predict.py --image ../data/processed/sample.jpg`

## Known limitations / next steps

- Trained on a very small sample (~15-20 images/class) as a
  proof-of-concept, not a clinically meaningful sample size.
- No cross-validation; Custom Vision's built-in train/test split was
  used as-is.
- With more time, next steps would be: a larger labeled sample, a
  custom PyTorch/TensorFlow model (e.g. fine-tuned ResNet) for more
  control over the architecture, and deployment via Azure ML managed
  endpoints instead of Custom Vision for more flexibility.
