# Retinal Image Classifier

A small proof-of-concept binary classifier for retinal fundus images
(No DR vs. Moderate DR), trained locally with PyTorch using transfer
learning on a pretrained ResNet18.

## Dataset

Sample images pulled from the [APTOS 2019 Blindness Detection]
(https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)
dataset on Kaggle (public, requires free Kaggle account). Only a
small subset (~20 images per class) was used here as a
proof-of-concept — not the full competition dataset, which is far
larger than needed for a demo like this.

Classes used: `No_DR` (no diabetic retinopathy) and `Moderate_DR`.

## Pipeline steps

1. **Collect** — small labeled subset selected from Kaggle via a
   Jupyter notebook (filtering `train.csv` by diagnosis label).
2. **Preprocess** — `preprocess.py` resizes, center-crops, and
   contrast-enhances each image.
3. **Train** — `train.py` fine-tunes a pretrained ResNet18's final
   layer on our two classes (transfer learning), with an 80/20
   train/validation split.
4. **Evaluate** — validation accuracy is printed each epoch during
   training.
5. **Predict** — `predict_local.py` loads the saved model and runs
   inference on a new image.

## How to reproduce

```
python retinal_classifier/preprocess.py --input_dir data/raw/No_DR --output_dir data/processed/No_DR
python retinal_classifier/preprocess.py --input_dir data/raw/Moderate_DR --output_dir data/processed/Moderate_DR
python -m retinal_classifier.train
python -m retinal_classifier.predict_local --image data/processed/No_DR/<some_image>.png
```

## Why transfer learning

With only ~40 labeled images total, training a CNN from scratch
would badly overfit. Freezing a pretrained backbone (already trained
on ImageNet) and only retraining the final layer lets the model
reuse general visual features instead of trying to learn them from
a tiny dataset.

## Known limitations / next steps

- Trained on a very small sample (~40 images total) as a
  proof-of-concept, not a clinically meaningful sample size — real
  validation accuracy on this scale should be read as noisy, not a
  reliable performance estimate.
- No k-fold cross-validation, no data augmentation yet.
- `predict.py` (kept in this folder) is a ready-to-use client for an
  Azure Custom Vision prediction endpoint. It wasn't used for this
  version — deployment went with a local PyTorch model instead — but
  it's a natural next step: fine-tune the full network (not just the
  last layer) on a larger sample, then deploy via Azure ML for a
  managed, scalable endpoint.
