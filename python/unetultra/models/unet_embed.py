"""UNetEmbed: UNetMulti with a frozen DINOv2 encoder fused in at the bottleneck.

DINOv2-small sees the same image resized to 448x448 and returns a 32x32 grid of 384-d patch
features, which is exactly the U-Net's bottleneck grid for a 512x512 input. Two things are
fused there: the patch features (projected down) and a similarity map against a class
prototype, the mean feature of the object in the training set. Everything DINO-specific,
including its normalisation and the prototype, lives inside the model, so from the outside it
has the same inputs and outputs as UNetMulti.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model import DoubleConv, Ups

DINO_NAME = "facebook/dinov2-small"
DINO_PATCH = 14
DINO_DIM = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenDino(nn.Module):
    """A frozen DINOv2 backbone at one fixed input size, returning a [B, C, g, g] feature grid.

    The position embeddings are interpolated once, here, and stored as a constant. That keeps
    the forward pass free of bicubic resizing, which Core ML cannot convert.
    """

    def __init__(self, backbone, input_size=448):
        super().__init__()
        if input_size % DINO_PATCH:
            raise ValueError(f"DINOv2 input size must be a multiple of {DINO_PATCH}")
        self.input_size = input_size
        self.grid = input_size // DINO_PATCH
        self.backbone = backbone
        embeddings = backbone.embeddings
        with torch.no_grad():
            dummy = torch.zeros(1, 1 + self.grid * self.grid, backbone.config.hidden_size)
            position = embeddings.interpolate_pos_encoding(dummy, input_size, input_size)
        self.register_buffer("position", position.detach().clone())
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)  # always in eval mode: the encoder is frozen

    def forward(self, image):
        """image: [B, 3, H, W] in 0..1. Returns patch features [B, C, grid, grid]."""
        x = F.interpolate(image, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        embeddings = self.backbone.embeddings
        patches = embeddings.patch_embeddings(x)
        cls = embeddings.cls_token.expand(patches.shape[0], -1, -1)
        tokens = torch.cat((cls, patches), dim=1) + self.position
        tokens = self.backbone.layernorm(self.backbone.encoder(tokens).last_hidden_state)
        grid = tokens[:, 1:, :].transpose(1, 2)
        return grid.reshape(grid.shape[0], grid.shape[1], self.grid, self.grid)


def load_dino(pretrained=True, name=DINO_NAME):
    """The DINOv2-small backbone. pretrained=False builds the same network untrained, with no
    download, for loading a checkpoint that already holds the weights (and for tests)."""
    from transformers import Dinov2Config, Dinov2Model

    if pretrained:
        return Dinov2Model.from_pretrained(name)
    return Dinov2Model(Dinov2Config(hidden_size=DINO_DIM, num_hidden_layers=12, num_attention_heads=6,
                                    mlp_ratio=4, patch_size=DINO_PATCH, image_size=518))


class UNetEmbed(nn.Module):
    """U-Net with FiLM metadata conditioning and frozen DINOv2 features at the bottleneck.

    Same interface as UNetMulti: forward(image, meta) -> (mask_logits, depth in metres).
    """

    def __init__(self, in_chan=3, features=(16, 32, 64, 128), n_meta=2, n_classes=1, backbone=None,
                 dino_input=448, embed_chan=64):
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
        self.dino = FrozenDino(backbone if backbone is not None else load_dino(pretrained=False), dino_input)
        dino_dim = self.dino.backbone.config.hidden_size
        self.embed = nn.Sequential(nn.Conv2d(dino_dim, embed_chan, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(embed_chan), nn.ReLU(inplace=True))
        # Mean object feature over the training set (unit length); zeros until compute_prototype runs.
        self.register_buffer("prototype", torch.zeros(dino_dim))
        self.fuse = DoubleConv(bottleneck_chan + embed_chan + 1, bottleneck_chan)
        for feature in reversed(features):
            self.ups.append(Ups(feature * 2, feature))
        self.meta = nn.Sequential(nn.Linear(n_meta, 64), nn.ReLU(), nn.Linear(64, 2 * bottleneck_chan))
        mask_channels = 1 if n_classes == 1 else n_classes + 1
        self.mask_head = nn.Conv2d(features[0], mask_channels, kernel_size=1)
        self.depth_head = nn.Conv2d(features[0], 1, kernel_size=1)

    def dino_features(self, image):
        """Unit-length DINOv2 patch features [B, C, g, g] (no gradient: the encoder is frozen)."""
        with torch.no_grad():
            return F.normalize(self.dino(image), dim=1)

    def forward(self, x, meta):
        patches = self.dino_features(x)
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        similarity = (patches * self.prototype.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        embedded = torch.cat((self.embed(patches), similarity), dim=1)
        if embedded.shape[2:] != x.shape[2:]:
            # Only when training at another size; at 512x512 with a 448 encoder both grids are 32x32.
            embedded = F.interpolate(embedded, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = self.fuse(torch.cat((x, embedded), dim=1))
        gamma, beta = self.meta(meta).chunk(2, dim=1)
        x = x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        for up in self.ups:
            x = up(x, skips.pop())
        return self.mask_head(x), F.softplus(self.depth_head(x))

    @torch.no_grad()
    def compute_prototype(self, dataset, device="cpu", max_samples=200):
        """Set the class prototype: the mean DINOv2 feature over object cells in a dataset.

        For several classes this is the mean over all object classes; a per-class prototype
        would be the natural extension. Returns the number of samples that contributed."""
        total = torch.zeros_like(self.prototype)
        used = 0
        for index in range(min(len(dataset), max_samples)):
            item = dataset[index]
            mask = ((item["labels"] > 0) & (item["labels"] != 255)).float()[None, None]
            if mask.sum() == 0:
                continue
            patches = self.dino_features(item["image"][None].to(device))[0]
            cells = F.adaptive_avg_pool2d(mask, patches.shape[1:])[0, 0].to(device)
            if cells.sum() <= 0:
                continue
            total += (patches * cells).sum(dim=(1, 2)) / cells.sum()
            used += 1
        if used:
            self.prototype.copy_(F.normalize(total / used, dim=0))
        return used
