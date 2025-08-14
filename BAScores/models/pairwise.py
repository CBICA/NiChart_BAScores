import torch
from torch import nn


class PairwiseModel3D(nn.Module):
    def __init__(self, encoder: nn.Module, device: str, meta: bool = False) -> None:
        super().__init__()
        self.meta = meta
        self.backbone = nn.Sequential(*list(encoder.net.children())[:-1]).to(device)
        if self.meta:
            self.linear = nn.LazyLinear(1, bias=False).to(device)
            in_1 = self.backbone(torch.randn(1, 1, 128, 128, 128).to(device))
            in_2 = self.backbone(torch.randn(1, 1, 128, 128, 128).to(device))
            in_3 = in_2 - in_1
            init_input = torch.cat((in_1, in_2, in_3), dim=1)
            self.linear(init_input)
        else:
            self.linear = nn.Sequential(*list(encoder.net.children())[-1]).to(device)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, return_features: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        f1 = self.backbone(x2)
        f2 = self.backbone(x1)
        f3 = f2 - f1
        if self.meta:
            features = torch.cat((f1, f2, f3), dim=1)
            res = self.linear(features)
        else:
            features = f3
            res = self.linear(f3)

        if return_features:
            return res, features

        return res
