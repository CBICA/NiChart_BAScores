from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F

from BAScores.models.cbam import CBAM
from BAScores.utils import init_weights


class LazyConv3d(nn.LazyConv3d):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        super(LazyConv3d, self).__init__(
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        weight_mean = (
            weight.mean(dim=1, keepdim=True)
            .mean(dim=2, keepdim=True)
            .mean(dim=3, keepdim=True)
            .mean(dim=4, keepdim=True)
        )
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv3d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class Residual(nn.Module):
    def __init__(
        self,
        num_channels: int,
        device: str,
        use_1x1conv: bool = False,
        strides: int = 1,
        cbam: bool = False,
    ) -> None:
        super().__init__()
        self.cbam = cbam
        self.conv1 = LazyConv3d(
            num_channels,
            kernel_size=3,
            padding=1,
            stride=strides,
            bias=False,
        ).to(device)

        self.conv2 = LazyConv3d(num_channels, kernel_size=3, padding=1, bias=False).to(
            device
        )
        if use_1x1conv:
            self.conv3 = LazyConv3d(num_channels, kernel_size=1, stride=strides).to(
                device
            )
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm3d().to(device)
        self.bn2 = nn.LazyBatchNorm3d().to(device)

        if cbam:
            self.cbam_module = CBAM(num_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.cbam:
            Y = self.cbam_module(Y)

        if self.conv3:
            X = self.conv3(X)

        Y += X
        return self.relu(Y)


class ResNet3D(nn.Module):
    def __init__(
        self,
        arch: Any,
        device: str,
        num_classes: int = 1,
        cbam: bool = False,
    ) -> None:
        super().__init__()

        def b1(in_channels: int) -> nn.Sequential:
            return nn.Sequential(
                LazyConv3d(
                    in_channels,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                nn.LazyBatchNorm3d(),
                nn.ReLU(inplace=True),
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
                        Residual(
                            num_channels, device, use_1x1conv=True, strides=2, cbam=cbam
                        )
                    )
                else:
                    blk.append(Residual(num_channels, device=device, cbam=cbam))
            return nn.Sequential(*blk).to(device)

        self.net = nn.Sequential(b1(arch[0][1])).to(device)

        for i, b in enumerate(arch):
            self.net.add_module(f"b{i+2}", block(*b, first_block=(i == 0)))  # type: ignore

        self.net.add_module(
            "features",
            nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
            ).to(device),
        )

        self.net.add_module(
            "final_layer",
            nn.Sequential(
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
