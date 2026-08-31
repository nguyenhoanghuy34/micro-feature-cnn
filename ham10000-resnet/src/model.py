import torch
import torch.nn as nn


# =========================================================
# 1. Basic Residual Block
# =========================================================

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()

        # -------------------------------------------------
        # Conv 1
        # -------------------------------------------------
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        # -------------------------------------------------
        # Conv 2
        # -------------------------------------------------
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # -------------------------------------------------
        # Shortcut
        # -------------------------------------------------

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm2d(out_channels)
            )


    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection
        out = out + identity

        out = self.relu(out)

        return out


# =========================================================
# 2. ResNet
# =========================================================

class ResNet(nn.Module):

    def __init__(
        self,
        block,
        layers,
        num_classes=7
    ):

        super().__init__()

        self.in_channels = 64

        # -------------------------------------------------
        # Stem
        # -------------------------------------------------

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # -------------------------------------------------
        # Residual stages
        # -------------------------------------------------

        self.layer1 = self._make_layer(
            block,
            64,
            layers[0],
            stride=1
        )

        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2
        )

        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2
        )

        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2
        )

        # -------------------------------------------------
        # Classification
        # -------------------------------------------------

        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.fc = nn.Linear(
            512 * block.expansion,
            num_classes
        )

        # -------------------------------------------------
        # Weight initialization
        # -------------------------------------------------

        self._initialize_weights()


    def _make_layer(
        self,
        block,
        out_channels,
        blocks,
        stride
    ):

        layers = []

        # First block
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride
            )
        )

        self.in_channels = out_channels

        # Remaining blocks
        for _ in range(1, blocks):

            layers.append(
                block(
                    self.in_channels,
                    out_channels
                )
            )

        return nn.Sequential(*layers)


    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(
                    m.weight,
                    1
                )

                nn.init.constant_(
                    m.bias,
                    0
                )


    def forward(self, x):

        # 224x224
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 112x112
        x = self.maxpool(x)

        # 56x56
        x = self.layer1(x)

        # 28x28
        x = self.layer2(x)

        # 14x14
        x = self.layer3(x)

        # 7x7
        x = self.layer4(x)

        # 1x1
        x = self.avgpool(x)

        x = torch.flatten(
            x,
            1
        )

        x = self.fc(x)

        return x


# =========================================================
# ResNet18
# =========================================================

def ResNet18(num_classes=7):

    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes
    )