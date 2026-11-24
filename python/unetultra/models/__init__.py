"""Model registry: one builder per architecture, selected by name.

Every architecture has the same outside shape, forward(image, meta) -> (mask_logits, depth),
so training, evaluation and export do not care which one they are given. To add one, write a
builder and register it in ARCHITECTURES.
"""

from ..model import UNetMulti
from .unet_embed import UNetEmbed, load_dino

STANDARD_META = ("log_focus_distance_m", "fx_norm")


def _unet_multi(n_classes, features, pretrained=True, **options):
    return UNetMulti(features=tuple(features), n_classes=n_classes)


def _unet_embed(n_classes, features, pretrained=True, dino_input=448, **options):
    return UNetEmbed(features=tuple(features), n_classes=n_classes, backbone=load_dino(pretrained),
                     dino_input=dino_input)


# name -> (builder, meta features the model is given, short description)
ARCHITECTURES = {
    "unet_multi": (_unet_multi, STANDARD_META, "U-Net with the focus input"),
    "image_only": (_unet_multi, (), "the same U-Net with no focus input (ablation)"),
    "unet_embed": (_unet_embed, STANDARD_META, "U-Net with the focus input and frozen DINOv2-small features"),
}


def build_model(architecture, n_classes=1, features=(16, 32, 64, 128), pretrained=True, **options):
    """Build a registered architecture. pretrained=False skips downloading frozen encoder weights,
    for when a checkpoint that already contains them is about to be loaded."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {architecture!r}; registered: {sorted(ARCHITECTURES)}")
    model = ARCHITECTURES[architecture][0](n_classes, features, pretrained=pretrained, **options)
    model.architecture = architecture
    return model


def meta_features(architecture):
    """Names of the meta features an architecture receives, in order ([] for an image-only model)."""
    return list(ARCHITECTURES[architecture][1])


def uses_meta(architecture):
    return bool(ARCHITECTURES[architecture][1])


__all__ = ["ARCHITECTURES", "STANDARD_META", "UNetEmbed", "UNetMulti", "build_model", "meta_features", "uses_meta"]
