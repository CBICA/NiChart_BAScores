import torch
from torch import nn

from BAScores.utils import init_weights


class AlexNet3D(nn.Module):
    def __init__(
        self,
        device: str,
        num_classes: int = 1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.LazyConv3d(
                96,
                kernel_size=(11, 11, 11),
                stride=(4, 4, 4),
                padding=(1, 1, 1),
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2)),
            nn.LazyConv3d(
                256,
                kernel_size=(5, 5, 5),
                padding=(2, 2, 2),
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2)),
            nn.LazyConv3d(
                384,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
            ),
            nn.ReLU(),
            nn.LazyConv3d(
                384,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
            ),
            nn.ReLU(),
            nn.LazyConv3d(
                256,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2)),
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes),
        ).to(device)

        # initialize lazy modules
        input_tensor = torch.randn(1, 1, 128, 128, 128).to(device)
        self.net(input_tensor)

        # initialize weights using kaiming
        init_weights(self.net)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)
