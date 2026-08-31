import torch

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from dataset import create_dataloaders
from model import SkinCancerCNN


DATA_DIR = "../data"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# Data
# =========================================================

_, _, test_loader, classes = \
    create_dataloaders(DATA_DIR)


# =========================================================
# Model
# =========================================================

model = SkinCancerCNN(
    num_classes=len(classes)
)

model.load_state_dict(
    torch.load(
        "../checkpoints/best_model.pth",
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()


# =========================================================
# Prediction
# =========================================================

all_labels = []
all_predictions = []
all_probabilities = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = outputs.argmax(
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )


# =========================================================
# Metrics
# =========================================================

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes
    )
)


print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)


auc = roc_auc_score(
    all_labels,
    all_probabilities
)

print(
    f"\nROC-AUC: {auc:.4f}"
)