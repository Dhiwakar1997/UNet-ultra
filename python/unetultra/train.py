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
from .model import UNetMulti
from .session import IGNORE_LABEL, Session


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 8
    lr: float = 1e-3
    depth_weight: float = 0.5
    # Weight of a second depth term averaged over object pixels only. The object covers about 1%
    # of the frame, so a plain L1 over all valid pixels (object_depth_weight = 0) barely sees it.
    object_depth_weight: float = 1.0
    features: tuple = (16, 32, 64, 128)
    use_meta: bool = True
    seed: int = 0


def split_sessions(paths, val_paths=None, val_fraction=0.25, seed=0):
    """Split session zips into train and validation by whole session, never by frame.

    Pass val_paths to choose the validation sessions yourself; otherwise sessions are shuffled
    and about val_fraction of them (at least one) are held out. Test sessions are refused."""
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
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


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
        for b in range(len(batch["image"])):
            for index in batch["instances"][b].unique().tolist():
                if index == 0:
                    continue
                region = (batch["instances"][b] == index) & batch["depth_valid"][b]
                if region.sum() >= 5:
                    predicted = pred_depth[b, 0][region].median()
                    errors.append(abs(float(predicted - batch["depth"][b][region].median())) * 100)
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
    n_classes = train_dataset.n_classes
    model = UNetMulti(features=tuple(config.features), n_classes=n_classes).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
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
                "classes": train_dataset.classes, "epoch": epoch, "val": row,
            }, best_path)
        log(f"{name} epoch {epoch:3d}  train {row['train_loss']:.4f}  val {row['val_loss']:.4f}  "
            f"dice {row['val_dice']:.3f}  distance err {row['val_distance_error_cm']:.2f} cm  "
            f"({time.time() - start:.0f}s){'  *' if score == best_score else ''}")
    (out_dir / f"{name}_history.json").write_text(json.dumps(history, indent=2))
    return history, best_path


def load_model(checkpoint_path, device="cpu"):
    """Load a checkpoint written by fit(). Returns (model in eval mode, checkpoint dict)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNetMulti(features=tuple(checkpoint["config"]["features"]), n_classes=checkpoint["n_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint
