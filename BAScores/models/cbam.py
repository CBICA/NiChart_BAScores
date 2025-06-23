import torch
import torch.nn.functional as F
from torch import nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


class CBAM(nn.Module):
    def __init__(self, channels: int, ratio: int = 4) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels, bias=False),
        ).to(device)

        self.sigmoid = nn.Sigmoid()
        self.spatial = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----- Channel Attention -----
        b, c, d, h, w = x.size()
        avg_pool = F.adaptive_avg_pool3d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool3d(x, 1).view(b, c)
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1, 1)
        x = x * channel_att

        # ----- Spatial Attention -----
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.sigmoid(self.spatial(spatial_att))
        x = x * spatial_att

        return x
