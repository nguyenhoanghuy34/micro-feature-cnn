import os

import torch
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import HAM10000Dataset
from model import ResNet18


# =========================================================
# Configuration
# =========================================================

CSV_PATH = "data/HAM10000_metadata.csv"

IMAGE_DIR = "data/HAM10000_images"

MODEL_PATH = "best_resnet18_ham10000.pth"

BATCH_SIZE = 32

NUM_CLASSES = 7


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# =========================================================
# Load dataset
# =========================================================

import pandas as pd

from sklearn.model_selection import train_test_split


df = pd.read_csv(
    CSV_PATH
)


# IMPORTANT:
# Must use the same split as training

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["dx"],
    random_state=42
)


# =========================================================
# Transform
# =========================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================================================
# Dataset
# =========================================================

val_dataset = HAM10000Dataset(
    val_df,
    IMAGE_DIR,
    transform
)


# =========================================================
# DataLoader
# =========================================================

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)


# =========================================================
# Model
# =========================================================

model = ResNet18(
    num_classes=NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# =========================================================
# Prediction
# =========================================================

all_labels = []

all_predictions = []


with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# =========================================================
# Metrics
# =========================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


print("\n==============================")
print("Evaluation Results")
print("==============================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


# =========================================================
# Classification Report
# =========================================================

class_names = val_dataset.classes


print("\n==============================")
print("Classification Report")
print("==============================")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# =========================================================
# Confusion Matrix
# =========================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)


fig, ax = plt.subplots(
    figsize=(9, 9)
)

disp.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title(
    "HAM10000 Confusion Matrix"
)

plt.tight_layout()


os.makedirs(
    "outputs",
    exist_ok=True
)

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.close()


print(
    "\nConfusion matrix saved to "
    "outputs/confusion_matrix.png"
)