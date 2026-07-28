"""
predict.py
----------
Sends a preprocessed retinal image to a deployed Azure Custom Vision
prediction endpoint and prints the returned class probabilities.

This is the "deploy and test on Azure" piece of the pipeline: model
training itself happens in the Custom Vision web portal (see
retinal_classifier/README.md for that walkthrough), and this script
is how our code integrates with the deployed model.

Setup:
  1. Set environment variables (do NOT hardcode keys in this file):
       set AZURE_CV_PREDICTION_KEY=your_key_here      (Windows cmd)
       $env:AZURE_CV_PREDICTION_KEY="your_key_here"   (PowerShell)
       set AZURE_CV_ENDPOINT_URL=https://<your-resource>.cognitiveservices.azure.com/customvision/v3.0/Prediction/<project-id>/classify/iterations/<iteration-name>/image
  2. python predict.py --image path/to/image.jpg
"""

import argparse
import os
import requests


def predict_image(image_path: str):
    endpoint_url = os.environ.get("AZURE_CV_ENDPOINT_URL")
    prediction_key = os.environ.get("AZURE_CV_PREDICTION_KEY")

    if not endpoint_url or not prediction_key:
        raise EnvironmentError(
            "Missing AZURE_CV_ENDPOINT_URL or AZURE_CV_PREDICTION_KEY. "
            "Set them as environment variables before running (see README)."
        )

    headers = {
        "Prediction-Key": prediction_key,
        "Content-Type": "application/octet-stream",
    }

    with open(image_path, "rb") as f:
        image_data = f.read()

    response = requests.post(endpoint_url, headers=headers, data=image_data)
    response.raise_for_status()
    result = response.json()

    print(f"\nPredictions for {image_path}:")
    for pred in sorted(result["predictions"], key=lambda p: p["probability"], reverse=True):
        print(f"  {pred['tagName']:<15} {pred['probability']*100:5.1f}%")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction against Azure Custom Vision.")
    parser.add_argument("--image", required=True, help="Path to a preprocessed retinal image")
    args = parser.parse_args()
    predict_image(args.image)
