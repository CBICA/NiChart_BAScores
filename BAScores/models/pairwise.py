import torch
from torch import nn


class PairwiseModel3D(nn.Module):
    def __init__(self, encoder: nn.Module, device: str) -> None:
        super().__init__()

        self.net = nn.ModuleList().to(device)
        self.net.append(encoder)
        self.net.append(nn.Linear(3, 1, bias=False).to(device))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.net[0](x1)
        x2 = self.net[0](x2)
        f = x2 - x1

        out = torch.cat((f, x1, x2), dim=1)
        return self.net[1](out)
