import os

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class HAM10000Dataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        transform=None
    ):

        self.df = dataframe.reset_index(drop=True)

        self.image_dir = image_dir

        self.transform = transform

        self.classes = sorted(
            self.df["dx"].unique()
        )

        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(self.classes)
        }


    def __len__(self):

        return len(self.df)


    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_id = row["image_id"]

        label = row["dx"]

        image_path = os.path.join(
            self.image_dir,
            image_id + ".jpg"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        label = self.class_to_idx[label]

        if self.transform:

            image = self.transform(image)

        return image, label