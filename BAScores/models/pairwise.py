import torch
from torch import nn


class PairwiseModel3D(nn.Module):
    def __init__(self, encoder: nn.Module, device: str) -> None:
        super().__init__()

        self.backbone = nn.Sequential(*list(encoder.net.children())[:-1]).to(device)
        # self.linear = nn.Sequential(*list(encoder.net.children())[-1]).to(device)
        self.linear = nn.LazyLinear(1, bias=False).to(device)

        in_1 = self.backbone(torch.randn(1, 1, 128, 128, 128).to(device))
        in_2 = self.backbone(torch.randn(1, 1, 128, 128, 128).to(device))
        in_3 = in_2 - in_1
        init_input = torch.cat((in_1, in_2, in_3), dim=1)
        self.linear(init_input)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        f1 = self.backbone(x2)
        f2 = self.backbone(x1)
        f3 = f2 - f1
        return self.linear(torch.cat((f1, f2, f3), dim=1))
