"""
predict_local.py
-----------------
Runs the locally trained model (from train.py) on a single image
and prints the predicted class with confidence.

Run with:  python -m retinal_classifier.predict_local --image path/to/image.png
"""

import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

MODEL_PATH = "retinal_classifier/model.pth"


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, classes


def predict(image_path: str):
    model, classes = load_model()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]

    print(f"\nPrediction for {image_path}:")
    for cls, prob in sorted(zip(classes, probs.tolist()), key=lambda x: x[1], reverse=True):
        print(f"  {cls:<15} {prob*100:5.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local model inference on a retinal image.")
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    predict(args.image)
