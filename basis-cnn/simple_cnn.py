import torch
import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # 28x28x1 -> 28x28x32
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            # 28x28x32 -> 14x14x32
            nn.MaxPool2d(kernel_size=2),

            # 14x14x32 -> 14x14x64
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            # 14x14x64 -> 7x7x64
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(

            # 7 * 7 * 64 = 3136
            nn.Linear(7 * 7 * 64, 128),

            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, start_dim=1)

        x = self.classifier(x)

        return x