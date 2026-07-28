"""
app.py
------
A small Streamlit web app that lets anyone upload a retinal fundus
image and see the trained model's prediction, without needing to
run any Python themselves.

Run locally with:  streamlit run app.py
Deployed for free via Streamlit Community Cloud (see README for steps).
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = "retinal_classifier/model.pth"


@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, classes


def predict(image: Image.Image, model, classes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]

    return sorted(zip(classes, probs.tolist()), key=lambda x: x[1], reverse=True)


st.set_page_config(page_title="Retinal Image Classifier", page_icon="👁️")
st.title("👁️ Retinal Image Classifier")
st.write(
    "Proof-of-concept classifier (No DR vs. Moderate DR) trained on a small "
    "sample from the APTOS 2019 dataset using transfer learning on ResNet18. "
    "**Not for medical use** — trained on ~40 images as a portfolio demo."
)

uploaded_file = st.file_uploader("Upload a retinal fundus image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width=300)

    model, classes = load_model()
    results = predict(image, model, classes)

    st.subheader("Prediction")
    for cls, prob in results:
        st.write(f"**{cls}**: {prob*100:.1f}%")
        st.progress(prob)
