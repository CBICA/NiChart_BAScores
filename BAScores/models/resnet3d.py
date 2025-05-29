from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def init_weights(net: nn.Module) -> None:
    for block in net:
        for layer in block.children():
            if isinstance(layer, nn.Conv3d) or isinstance(layer, nn.LazyConv3d):
                for name, weight in layer.named_parameters():
                    if "weight" in name:
                        nn.init.kaiming_normal_(weight)
                    if "bias" in name:
                        nn.init.constant_(weight, 0.0)


class Residual(nn.Module):
    def __init__(
        self,
        num_channels: int,
        device: str,
        use_1x1conv: bool = False,
        strides: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.LazyConv3d(
            num_channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            stride=strides,
            bias=False,
        ).to(device)
        self.conv2 = nn.LazyConv3d(
            num_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False
        ).to(device)
        if use_1x1conv:
            self.conv3 = nn.LazyConv3d(
                num_channels, kernel_size=(1, 1, 1), stride=strides
            ).to(device)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm3d().to(device)
        self.bn2 = nn.LazyBatchNorm3d().to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            X = self.conv3(X)

        Y += X
        return F.relu(Y)


class ResNet3D(nn.Module):
    def __init__(self, arch: Any, device: str, num_classes: int = 1) -> None:
        super().__init__()

        def b1(in_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LazyConv3d(
                    in_channels,
                    kernel_size=(7, 7, 7),
                    stride=(2, 2, 2),
                    padding=(3, 3, 3),
                    bias=False,
                ),
                nn.LazyBatchNorm3d(),
                nn.ReLU(),
                nn.MaxPool3d(
                    kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
                ),
            ).to(device)

        def block(
            num_residuals: int, num_channels: int, first_block: bool = False
        ) -> nn.Sequential:
            blk = []
            for i in range(num_residuals):
                if i == 0 and not first_block:
                    blk.append(
                        Residual(num_channels, device, use_1x1conv=True, strides=2)
                    )
                else:
                    blk.append(Residual(num_channels, device=device))
            return nn.Sequential(*blk).to(device)

        self.net = nn.Sequential(b1(arch[0][1])).to(device)

        for i, b in enumerate(arch):
            self.net.add_module(f"b{i+2}", block(*b, first_block=(i == 0)))  # type: ignore

        self.net.add_module(
            "last",
            nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.LazyLinear(num_classes),
            ).to(device),
        )

        # initialize the lazy modules
        init_input = torch.randn(1, 1, 128, 128, 128).to(device)
        self.net(init_input)

        # initialize weights using kaiming
        init_weights(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
