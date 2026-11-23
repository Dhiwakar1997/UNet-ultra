"""UNetMulti: a small U-Net with FiLM conditioning on camera metadata and two heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

INPUT_WIDTH = 512
INPUT_HEIGHT = 512  # the centre square of the frame, resized (see geometry.CenterSquareCrop)


class DoubleConv(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Ups(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_chan * 2, out_chan)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape != skip.shape:
            # Never triggers at the fixed 512x512 export size; kept for training flexibility.
            x = TF.resize(x, size=skip.shape[2:])
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class UNetMulti(nn.Module):
    """U-Net with FiLM conditioning on camera metadata at the bottleneck, and two heads:
    mask (1 channel if n_classes == 1, else n_classes + 1 including background) and depth in metres.

    meta is [log(calibrated focus distance in m), fx_scaled / 1000 - 1]. It never contains depth.
    """

    def __init__(self, in_chan=3, features=(16, 32, 64, 128), n_meta=2, n_classes=1):
        super().__init__()
        self.n_classes = n_classes
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(DoubleConv(in_chan, feature))
            in_chan = feature
        bottleneck_chan = features[-1] * 2
        self.bottleneck = DoubleConv(features[-1], bottleneck_chan)
        for feature in reversed(features):
            self.ups.append(Ups(feature * 2, feature))
        self.meta = nn.Sequential(nn.Linear(n_meta, 64), nn.ReLU(), nn.Linear(64, 2 * bottleneck_chan))
        mask_channels = 1 if n_classes == 1 else n_classes + 1
        self.mask_head = nn.Conv2d(features[0], mask_channels, kernel_size=1)
        self.depth_head = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x, meta):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        gamma, beta = self.meta(meta).chunk(2, dim=1)
        x = x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        for up in self.ups:
            x = up(x, skips.pop())
        return self.mask_head(x), F.softplus(self.depth_head(x))
