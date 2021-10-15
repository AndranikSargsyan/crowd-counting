import torch.nn as nn
from torch import Tensor
from torchvision import models


class CSRNet(nn.Module):
    def __init__(self) -> None:
        super(CSRNet, self).__init__()

        backbone = models.vgg16(pretrained=True)
        self.frontend = nn.Sequential(*list(backbone.features.children())[:22])  # size=(N, 512, x.H/8, x.W/8)
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=2, dilation=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=2, dilation=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=2, dilation=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=(3, 3), padding=2, dilation=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(3, 3), padding=2, dilation=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=2, dilation=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=(1, 1))
        )

        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        return x
