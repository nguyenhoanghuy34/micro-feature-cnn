import random
import numpy as np
import torch
import matplotlib.pyplot as plt


# =========================================================
# 1. Reproducibility
# =========================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# =========================================================
# 2. Save checkpoint
# =========================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    path
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss
    }

    torch.save(
        checkpoint,
        path
    )


# =========================================================
# 3. Load checkpoint
# =========================================================

def load_checkpoint(
    model,
    optimizer,
    path,
    device
):

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint


# =========================================================
# 4. Plot training history
# =========================================================

def plot_training_history(
    history,
    save_path="outputs/training_curve.png"
):

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )


    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training and Validation Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()


    # -----------------------------------------------------
    # Accuracy
    # -----------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_acc"],
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        history["val_acc"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Training and Validation Accuracy")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/accuracy_curve.png",
        dpi=300
    )

    plt.close()