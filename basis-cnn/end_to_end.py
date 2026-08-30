import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =========================================================
# 1. Device
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# =========================================================
# 2. Dataset
# =========================================================

transform = transforms.ToTensor()


train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# =========================================================
# 3. DataLoader
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))


# =========================================================
# 4. CNN Model
# =========================================================

class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # 1 × 28 × 28
            # ↓
            # 32 × 28 × 28
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            # 32 × 28 × 28
            # ↓
            # 32 × 14 × 14
            nn.MaxPool2d(kernel_size=2),

            # 32 × 14 × 14
            # ↓
            # 64 × 14 × 14
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            # 64 × 14 × 14
            # ↓
            # 64 × 7 × 7
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(

            # 64 × 7 × 7 = 3136
            nn.Linear(7 * 7 * 64, 128),

            nn.ReLU(),

            # 128 → 10 classes
            nn.Linear(128, 10)
        )

    def forward(self, x):

        x = self.features(x)

        # [batch, 64, 7, 7]
        # ↓
        # [batch, 3136]
        x = torch.flatten(x, start_dim=1)

        x = self.classifier(x)

        return x


# =========================================================
# 5. Create Model
# =========================================================

model = SimpleCNN().to(device)

print(model)


# =========================================================
# 6. Loss Function
# =========================================================

criterion = nn.CrossEntropyLoss()


# =========================================================
# 7. Optimizer
# =========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# =========================================================
# 8. Training
# =========================================================

num_epochs = 5


for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # ---------------------------------------------
        # Forward Pass
        # ---------------------------------------------

        outputs = model(images)

        # ---------------------------------------------
        # Calculate Loss
        # ---------------------------------------------

        loss = criterion(outputs, labels)

        # ---------------------------------------------
        # Backpropagation
        # ---------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ---------------------------------------------
        # Accumulate loss
        # ---------------------------------------------

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Loss: {average_loss:.4f}"
    )


# =========================================================
# 9. Evaluation
# =========================================================

model.eval()

correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # Get predicted class
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


accuracy = 100 * correct / total


print(f"Test Accuracy: {accuracy:.2f}%")


# =========================================================
# 10. Save Model
# =========================================================

torch.save(
    model.state_dict(),
    "simple_cnn_mnist.pth"
)

print("Model saved!")