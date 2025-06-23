import torch
from torch import nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


class CBAM(nn.Module):
    def __init__(self, channels: int, ratio: int = 4) -> None:
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels),
        ).to(device)

        self.conv1 = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False).to(device)

        self.max_pool = nn.AdaptiveMaxPool3d((1, 1, 1)).to(device)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1)).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc_max = self.max_pool(x)
        fc_max = fc_max.view(fc_max.size(0), -1)
        fc_avg = self.avg_pool(x)
        fc_avg = fc_avg.view(fc_avg.size(0), -1)

        fc_max = self.shared_mlp(fc_max)
        fc_avg = self.shared_mlp(fc_avg)

        cbam_features = torch.add(fc_avg, fc_max)
        Mc = nn.Sigmoid()(cbam_features)
        Mc = Mc.view(Mc.size(0), Mc.size(1), 1, 1, 1)
        Mf1 = Mc * x

        max_pool = torch.max(Mf1, 1)[0].unsqueeze(1)
        avg_pool = torch.mean(Mf1, 1).unsqueeze(1)
        Ms = torch.cat((max_pool, avg_pool), dim=1)

        Ms = self.conv1(Ms)
        Ms = nn.Sigmoid()(Ms)
        Ms = Ms.view(Ms.size(0), 1, Ms.size(2), Ms.size(3), Ms.size(4))
        Mf2 = Ms * Mf1
        return Mf2
