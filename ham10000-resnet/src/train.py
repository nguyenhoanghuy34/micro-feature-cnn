import os

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import (
    HAM10000Dataset
)

from model import (
    ResNet18
)

from torchvision import transforms


# =========================================================
# Configuration
# =========================================================

CSV_PATH = "data/HAM10000_metadata.csv"

IMAGE_DIR = "data/HAM10000_images"

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

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
# Load metadata
# =========================================================

df = pd.read_csv(
    CSV_PATH
)

print(
    "Total images:",
    len(df)
)


# =========================================================
# Train / Validation split
# =========================================================

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["dx"],
    random_state=42
)


print(
    "Train:",
    len(train_df)
)

print(
    "Validation:",
    len(val_df)
)


# =========================================================
# Transform
# =========================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(20),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

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

train_dataset = HAM10000Dataset(
    train_df,
    IMAGE_DIR,
    train_transform
)

val_dataset = HAM10000Dataset(
    val_df,
    IMAGE_DIR,
    val_transform
)


# =========================================================
# DataLoader
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# =========================================================
# Model
# =========================================================

model = ResNet18(
    num_classes=NUM_CLASSES
)

model = model.to(device)


# =========================================================
# Loss
# =========================================================

criterion = nn.CrossEntropyLoss()


# =========================================================
# Optimizer
# =========================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# =========================================================
# Scheduler
# =========================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    patience=2,

    factor=0.5
)


# =========================================================
# Training
# =========================================================

best_val_loss = float("inf")


for epoch in range(EPOCHS):

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        # Clear gradients
        optimizer.zero_grad()


        # Forward
        outputs = model(images)


        # Loss
        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        # Statistics
        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss /
        len(train_loader)
    )

    train_acc = (
        correct /
        total
    )


    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    model.eval()

    val_loss = 0.0

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            val_loss += loss.item()


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


    val_loss /= len(val_loader)

    val_acc = correct / total


    # -----------------------------------------------------
    # Scheduler
    # -----------------------------------------------------

    scheduler.step(
        val_loss
    )


    # -----------------------------------------------------
    # Save best model
    # -----------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "best_resnet18_ham10000.pth"
        )


    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.4f} "
        f"LR: {current_lr:.6f}"
    )