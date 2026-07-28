"""
train.py
--------
Trains a small retinal image classifier using transfer learning:
we take a ResNet18 already pretrained on ImageNet, freeze most of
its layers, and only retrain the final classification layer on our
two classes (No_DR / Moderate_DR).

Why transfer learning instead of training from scratch: with only
~40 labeled images, a model trained from scratch would badly
overfit. A pretrained backbone already knows general visual features
(edges, textures, shapes) from millions of images, so we only need
to teach it the final step: mapping those features to our two
specific classes.

Expects images laid out like:
    data/processed/No_DR/*.png
    data/processed/Moderate_DR/*.png
(this is exactly what preprocess.py produces)

Run with:  python -m retinal_classifier.train
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


DATA_DIR = "data/processed"
MODEL_OUT = "retinal_classifier/model.pth"
NUM_EPOCHS = 8
BATCH_SIZE = 8
LEARNING_RATE = 0.001


def get_dataloaders():
    # ImageNet normalization stats -- required since we're using an
    # ImageNet-pretrained backbone, which expects inputs normalized
    # the same way its original training data was.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    print(f"Classes found: {full_dataset.classes}")
    print(f"Total images: {len(full_dataset)}")

    # Small dataset -> simple 80/20 split rather than a separate val script
    val_size = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, full_dataset.classes


def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze all pretrained layers -- we don't want to update them,
    # only the new final layer we're about to add.
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully-connected layer to output our number
    # of classes instead of ImageNet's 1000. This new layer is the
    # only part that actually gets trained.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader, val_loader, classes = get_dataloaders()
    model = build_model(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    # Only the new final layer has requires_grad=True, so this only
    # optimizes that layer -- the pretrained backbone stays frozen.
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        # Quick validation pass each epoch
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}  train_loss={avg_loss:.4f}  val_acc={val_acc:.2%}")

    torch.save({"model_state": model.state_dict(), "classes": classes}, MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")


if __name__ == "__main__":
    train()
