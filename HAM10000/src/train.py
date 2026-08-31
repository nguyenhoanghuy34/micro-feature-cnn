import os

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from dataset import create_dataloaders
from model import SkinCancerCNN


# =========================================================
# Configuration
# =========================================================

DATA_DIR = "../data"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# Data
# =========================================================

train_loader, val_loader, test_loader, classes = \
    create_dataloaders(DATA_DIR)

print("Classes:", classes)
print("Device:", DEVICE)


# =========================================================
# Model
# =========================================================

model = SkinCancerCNN(
    num_classes=len(classes)
)

model = model.to(DEVICE)


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
    weight_decay=1e-4
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

    train_loss = 0
    train_correct = 0
    train_total = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, labels in progress_bar:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()

        # Update weights
        optimizer.step()

        # Metrics
        train_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    train_loss /= len(train_loader)

    train_accuracy = (
        train_correct / train_total
    )


    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss /= len(val_loader)

    val_accuracy = (
        val_correct / val_total
    )


    # -----------------------------------------------------
    # Scheduler
    # -----------------------------------------------------

    scheduler.step(val_loss)


    print(
        f"\n"
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f}\n"
        f"Val Loss:   {val_loss:.4f} | "
        f"Val Acc:   {val_accuracy:.4f}"
    )


    # -----------------------------------------------------
    # Save best model
    # -----------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        os.makedirs(
            "../checkpoints",
            exist_ok=True
        )

        torch.save(
            model.state_dict(),
            "../checkpoints/best_model.pth"
        )

        print("Saved best model.")