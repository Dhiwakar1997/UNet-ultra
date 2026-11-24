"""Losses, the training loop and checkpoint handling (used by 02_train.ipynb)."""

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .masks import pick_device
from .models import build_model
from .session import IGNORE_LABEL, Session


@dataclass
class TrainConfig:
    # A name from unetultra.models.ARCHITECTURES: "unet_multi", "image_only" or "unet_embed".
    architecture: str = "unet_multi"
    epochs: int = 60
    batch_size: int = 8
    lr: float = 1e-3
    depth_weight: float = 0.5
    # Weight of a second depth term averaged over object pixels only. The object covers about 1%
    # of the frame, so a plain L1 over all valid pixels (object_depth_weight = 0) barely sees it.
    object_depth_weight: float = 1.0
    features: tuple = (16, 32, 64, 128)
    use_meta: bool = True  # kept for old checkpoints; the architecture decides this now
    seed: int = 0
    # Extra options for the architecture's builder, e.g. {"dino_input": 448} for unet_embed.
    options: dict = None


def split_sessions(paths, val_paths=None, val_fraction=0.25, seed=0):
    """Split session zips into train and validation by whole session, never by frame.

    Pass val_paths to choose the validation sessions yourself; otherwise sessions are shuffled
    and about val_fraction of them (at least one) are held out. Test sessions are refused.
    With a single training session this raises; use split_single_session instead."""
    paths = [str(p) for p in paths]
    for path in paths + [str(p) for p in (val_paths or [])]:
        with Session(path) as sess:
            if sess.split != "train":
                raise ValueError(f"{path} is a '{sess.split}' session; it must not be used for training")
    if val_paths:
        val = [str(p) for p in val_paths]
        return [p for p in paths if p not in val], val
    if len(paths) < 2:
        raise ValueError(
            "Need at least two training sessions: one whole session is held out for validation, "
            "because frames from the same session are near-duplicates."
        )
    order = paths[:]
    random.Random(seed).shuffle(order)
    n_val = max(1, round(len(order) * val_fraction))
    return sorted(order[n_val:]), sorted(order[:n_val])


def split_single_session(dataset, val_fraction=0.2):
    """Fallback for one training session: hold out its last block of samples for validation.

    Returns (train_indices, val_indices) into `dataset`. The held-out samples are the last
    ones captured, as one contiguous block, never a random pick: neighbouring frames are
    near-duplicates, so a random split would leak. Even so, the validation numbers come from
    the same room and day as training and will look better than the truth; only the separate
    test session gives an honest score."""
    n = len(dataset)
    if n < 8:
        raise ValueError(f"only {n} usable samples; need at least 8 to hold some out for validation")
    # Object samples and no-object samples are held out separately (the last block of each), so
    # training still gets no-object samples even if they were all captured at the end.
    background = [bool(dataset.sample_meta(i)["background_only"]) if hasattr(dataset, "sample_meta") else False
                  for i in range(n)]
    train, val = [], []
    for group in ([i for i in range(n) if not background[i]], [i for i in range(n) if background[i]]):
        n_val = max(1, round(len(group) * val_fraction)) if len(group) >= 2 else 0
        n_val = max(n_val, 2) if group and not background[group[0]] else n_val
        train += group[: len(group) - n_val]
        val += group[len(group) - n_val :]
    return sorted(train), sorted(val)


def soft_dice_loss(probability, target, weight):
    """1 - soft Dice over a whole batch; `weight` zeroes ignored pixels. All shapes [B, H, W]."""
    probability, target = probability * weight, target * weight
    intersection = (probability * target).sum()
    return 1 - (2 * intersection + 1) / (probability.sum() + target.sum() + 1)


def mask_loss(logits, labels, n_classes):
    """BCE + soft Dice for one class; cross-entropy + mean per-class soft Dice for several."""
    keep = (labels != IGNORE_LABEL).float()
    if n_classes == 1:
        target = (labels == 1).float()
        bce = F.binary_cross_entropy_with_logits(logits[:, 0], target, weight=keep, reduction="sum")
        return bce / keep.sum().clamp(min=1) + soft_dice_loss(torch.sigmoid(logits[:, 0]), target, keep)
    ce = F.cross_entropy(logits, labels, ignore_index=IGNORE_LABEL)
    probability = logits.softmax(dim=1)
    dice = [soft_dice_loss(probability[:, c], (labels == c).float(), keep) for c in range(logits.shape[1])]
    return ce + torch.stack(dice).mean()


def depth_loss(pred, depth, valid, labels, object_weight=1.0):
    """L1 on valid depth pixels, plus object_weight x the L1 averaged over object pixels alone.

    The second term gives the object's depth a say that does not shrink with its pixel count."""
    error = (pred[:, 0] - depth).abs()
    everywhere = valid.float()
    loss = (error * everywhere).sum() / everywhere.sum().clamp(min=1)
    if object_weight:
        on_object = (valid & (labels > 0) & (labels != IGNORE_LABEL)).float()
        if on_object.sum() > 0:
            loss = loss + object_weight * (error * on_object).sum() / on_object.sum()
    return loss


def total_loss(mask_logits, pred_depth, batch, n_classes, config):
    m = mask_loss(mask_logits, batch["labels"], n_classes)
    d = depth_loss(pred_depth, batch["depth"], batch["depth_valid"], batch["labels"], config.object_depth_weight)
    return m + config.depth_weight * d, m, d


def _to_device(batch, device):
    """Move a batch's tensors to the device. float64 becomes float32 first: MPS has no float64."""
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.dtype == torch.float64:
                value = value.float()
            value = value.to(device)
        moved[key] = value
    return moved


def foreground(mask_logits, n_classes):
    """Predicted class id per pixel [B, H, W] (0 = background)."""
    if n_classes == 1:
        return (mask_logits[:, 0] > 0).long()
    return mask_logits.argmax(dim=1)


@torch.no_grad()
def validate(model, loader, device, config):
    """Validation loss, foreground Dice, and object-distance error in cm.

    Distance error compares the median predicted depth with the median LiDAR depth over each
    true object's pixels, so it is defined even before the mask head has learned anything."""
    model.eval()
    losses, errors, intersection, total = [], [], 0.0, 0.0
    for batch in loader:
        batch = _to_device(batch, device)
        mask_logits, pred_depth = model(batch["image"], batch["meta"])
        losses.append(float(total_loss(mask_logits, pred_depth, batch, model.n_classes, config)[0]))
        pred_fg = foreground(mask_logits, model.n_classes) > 0
        true_fg = (batch["labels"] > 0) & (batch["labels"] != IGNORE_LABEL)
        intersection += float((pred_fg & true_fg).sum())
        total += float(pred_fg.sum() + true_fg.sum())
        # Per-object statistics on the CPU: unique and median are not reliable on every backend.
        instances, valid = batch["instances"].cpu(), batch["depth_valid"].cpu()
        predicted_depth, true_depth = pred_depth[:, 0].float().cpu(), batch["depth"].cpu()
        for b in range(len(instances)):
            for index in instances[b].unique().tolist():
                if index == 0:
                    continue
                region = (instances[b] == index) & valid[b]
                if region.sum() >= 5:
                    predicted = predicted_depth[b][region].median()
                    errors.append(abs(float(predicted - true_depth[b][region].median())) * 100)
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_dice": 2 * intersection / total if total else 1.0,
        "val_distance_error_cm": float(np.mean(errors)) if errors else float("nan"),
    }


def fit(train_dataset, val_dataset, config, out_dir, name="model", device=None, log=print):
    """Train UNetMulti and keep the checkpoint with the lowest validation object-distance error.

    Returns (history, best_checkpoint_path). History is a list of per-epoch dicts and is also
    written next to the checkpoint as <name>_history.json."""
    device = device or pick_device()
    torch.manual_seed(config.seed)
    source = getattr(train_dataset, "dataset", train_dataset)  # a Subset keeps the real dataset here
    n_classes = source.n_classes
    model = build_model(config.architecture, n_classes, config.features, **(config.options or {})).to(device)
    if hasattr(model, "compute_prototype"):
        used = model.compute_prototype(source, device)
        log(f"{name}: class prototype from {used} training samples")
    optimiser = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs, eta_min=config.lr * 0.05)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator,
                              drop_last=len(train_dataset) > config.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path, best_score, history = out_dir / f"{name}.pt", float("inf"), []

    for epoch in range(1, config.epochs + 1):
        model.train()
        start, sums = time.time(), np.zeros(3)
        for batch in train_loader:
            batch = _to_device(batch, device)
            mask_logits, pred_depth = model(batch["image"], batch["meta"])
            loss, m, d = total_loss(mask_logits, pred_depth, batch, n_classes, config)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            sums += [loss.item(), m.item(), d.item()]
        scheduler.step()
        row = {"epoch": epoch, "train_loss": sums[0] / len(train_loader), "train_mask_loss": sums[1] / len(train_loader),
               "train_depth_loss": sums[2] / len(train_loader), **validate(model, val_loader, device, config)}
        history.append(row)
        score = row["val_distance_error_cm"] if np.isfinite(row["val_distance_error_cm"]) else row["val_loss"]
        if score < best_score:
            best_score = score
            torch.save({
                "state_dict": model.state_dict(), "config": asdict(config), "n_classes": n_classes,
                "architecture": config.architecture, "sessions": sorted({Path(p).name for p in source.paths}),
                "classes": source.classes, "epoch": epoch, "val": row,
            }, best_path)
        log(f"{name} epoch {epoch:3d}  train {row['train_loss']:.4f}  val {row['val_loss']:.4f}  "
            f"dice {row['val_dice']:.3f}  distance err {row['val_distance_error_cm']:.2f} cm  "
            f"({time.time() - start:.0f}s){'  *' if score == best_score else ''}")
    (out_dir / f"{name}_history.json").write_text(json.dumps(history, indent=2))
    return history, best_path


def load_model(checkpoint_path, device="cpu"):
    """Load a checkpoint written by fit(). Returns (model in eval mode, checkpoint dict)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    # Checkpoints from before the registry are UNetMulti, with or without the focus input.
    checkpoint.setdefault("architecture", "unet_multi" if config.get("use_meta", True) else "image_only")
    options = {k: v for k, v in (config.get("options") or {}).items() if k != "pretrained"}
    # pretrained=False: the checkpoint already holds any frozen encoder's weights, so nothing is downloaded.
    model = build_model(checkpoint["architecture"], checkpoint["n_classes"], config["features"],
                        pretrained=False, **options)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint
