from torch.utils.data import Dataset
from PIL import Image
import torch
import os


class SegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        name = self.images[idx]

        image = Image.open(
            os.path.join(self.image_dir, name)
        ).convert("L")

        mask = Image.open(
            os.path.join(self.mask_dir, name)
        ).convert("L")

        image = torch.tensor(
            __import__("numpy").array(image),
            dtype=torch.float32
        ).unsqueeze(0) / 255.0

        mask = torch.tensor(
            __import__("numpy").array(mask),
            dtype=torch.float32
        ).unsqueeze(0) / 255.0

        return image, mask