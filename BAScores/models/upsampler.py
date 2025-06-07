import torch
from torch import nn


class Up(nn.Module):
    def __init__(
        self,
        scale_factor: int,
        hidden: int,
        out: int,
        device: str,
    ) -> None:
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=scale_factor, mode="trilinear", align_corners=True
        ).to(device)
        self.conv = nn.Sequential(
            nn.LazyConv3d(hidden, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.LazyBatchNorm3d(),
            nn.ReLU(inplace=True),
            nn.LazyConv3d(out, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.LazyBatchNorm3d(),
            nn.ReLU(inplace=True),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(x)


class FeatureDecoder(nn.Module):
    def __init__(
        self,
        device: str,
    ) -> None:
        super().__init__()

        self.decoder = nn.Sequential(
            Up(scale_factor=2, hidden=256, out=128, device=device),
            Up(scale_factor=2, hidden=128, out=64, device=device),
            Up(scale_factor=2, hidden=64, out=32, device=device),
            Up(scale_factor=2, hidden=32, out=16, device=device),
            Up(scale_factor=2, hidden=16, out=8, device=device),
        )
        self.final_conv = nn.LazyConv3d(1, kernel_size=1).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_conv(self.decoder(x))
