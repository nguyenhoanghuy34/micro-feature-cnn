import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.unet import UNet
from dataset import SegmentationDataset


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = SegmentationDataset(
    "data/images",
    "data/masks"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

model = UNet().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


for epoch in range(10):

    model.train()

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(outputs, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(
        f"Epoch {epoch+1}/10 - Loss: {loss.item():.4f}"
    )