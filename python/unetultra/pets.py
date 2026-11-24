"""Side experiment: plain U-Net against U-Net + DINOv2 on a public segmentation dataset.

Uses the Oxford-IIIT Pet dataset (about 7,400 photos of cats and dogs with a pixel mask each).
There is no depth, focus or focal length here, so only the mask head is trained and the meta
input is zero: this isolates one question, whether frozen DINOv2 features help the U-Net
segment. Used by notebooks/06_pets_unet_vs_dino.ipynb.
"""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .masks import pick_device
from .models import build_model
from .session import IGNORE_LABEL
from .train import foreground, mask_loss

# Oxford-IIIT Pet trimap values.
TRIMAP_PET, TRIMAP_BACKGROUND, TRIMAP_BORDER = 1, 2, 3
ARCHITECTURES = {"unet_multi": "U-Net", "unet_embed": "U-Net + DINOv2"}


def load_oxford_pets(root, split):
    """The Oxford-IIIT Pet dataset through torchvision; downloads about 800 MB on first use.
    split is "trainval" (3,680 images) or "test" (3,669 images). Items are (image, trimap)."""
    from torchvision.datasets import OxfordIIITPet

    return OxfordIIITPet(str(root), split=split, target_types="segmentation", download=True)


def trimap_to_labels(trimap):
    """Trimap -> training labels: 1 pet, 0 background, 255 (ignored) for the uncertain border."""
    trimap = np.asarray(trimap)
    labels = np.full(trimap.shape, IGNORE_LABEL, dtype=np.uint8)
    labels[trimap == TRIMAP_PET] = 1
    labels[trimap == TRIMAP_BACKGROUND] = 0
    return labels


class PetSegmentation(Dataset):
    """(image, trimap) pairs resized to a square, as the dicts the models and losses expect.

    `base` is anything indexable that returns (PIL image, PIL trimap). Images are squashed to
    size x size (bilinear); trimaps use nearest neighbour so no label values are invented."""

    def __init__(self, base, indices=None, size=256, augment=False, seed=0):
        self.base = base
        self.indices = list(range(len(base))) if indices is None else list(indices)
        self.size = size
        self.augment = augment
        self.generator = torch.Generator().manual_seed(seed)
        self._cache = {}

    def __len__(self):
        return len(self.indices)

    def _load(self, position):
        image, trimap = self.base[self.indices[position]]
        image = np.asarray(image.convert("RGB").resize((self.size, self.size), Image.BILINEAR))
        labels = trimap_to_labels(trimap.resize((self.size, self.size), Image.NEAREST))
        return image, labels

    def __getitem__(self, position):
        if position not in self._cache:
            self._cache[position] = self._load(position)
        image, labels = self._cache[position]
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        labels = torch.from_numpy(labels.astype(np.int64))
        if self.augment:
            def rand(lo, hi):
                return float(torch.empty(1).uniform_(lo, hi, generator=self.generator))

            image = image * rand(0.8, 1.2)
            mean = image.mean()
            image = ((image - mean) * rand(0.8, 1.2) + mean).clamp(0, 1)
            if rand(0, 1) < 0.5:
                image, labels = image.flip(-1), labels.flip(-1)
        return {"image": image, "labels": labels, "meta": torch.zeros(2), "index": position}


def split_indices(n, train_limit=None, val_size=300, seed=0):
    """Shuffle 0..n-1 once and cut it into (train, validation) index lists."""
    order = np.random.default_rng(seed).permutation(n).tolist()
    val = order[:val_size]
    train = order[val_size:]
    return (train[:train_limit] if train_limit else train), val


def build(architecture, size, pretrained=True):
    """One of the two models at a given input size. For the DINOv2 model the encoder input is
    chosen so its patch grid equals the U-Net bottleneck grid (size / 16)."""
    if architecture == "unet_embed":
        return build_model(architecture, n_classes=1, pretrained=pretrained, dino_input=size // 16 * 14)
    return build_model(architecture, n_classes=1)


def count_parameters(model):
    """(trainable, total) parameter counts."""
    return (sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters()))


def dice_iou_per_image(pred, labels):
    """Dice and IoU per image over the labelled pixels. pred, labels: [B, H, W]. Returns two lists."""
    keep = labels != IGNORE_LABEL
    pred, true = (pred > 0) & keep, (labels == 1) & keep
    intersection = (pred & true).flatten(1).sum(1).float()
    total = pred.flatten(1).sum(1).float() + true.flatten(1).sum(1).float()
    dice = torch.where(total > 0, 2 * intersection / total.clamp(min=1), torch.ones_like(total))
    union = total - intersection
    iou = torch.where(union > 0, intersection / union.clamp(min=1), torch.ones_like(union))
    return dice.tolist(), iou.tolist()


@torch.no_grad()
def evaluate_segmentation(model, dataset, device=None, batch_size=16):
    """Mean Dice and IoU over a dataset, per-image scores, predictions and latency per image."""
    device = device or pick_device()
    model = model.to(device).eval()
    dice, iou, predictions, seconds, images = [], [], [], 0.0, 0
    for batch in DataLoader(dataset, batch_size=batch_size):
        image, meta = batch["image"].to(device), batch["meta"].to(device)
        started = time.perf_counter()
        logits, _ = model(image, meta)
        pred = foreground(logits, 1).cpu()
        seconds += time.perf_counter() - started
        images += len(image)
        d, i = dice_iou_per_image(pred, batch["labels"])
        dice += d
        iou += i
        predictions += [p.numpy().astype(bool) for p in pred]
    return {"dice": float(np.mean(dice)), "iou": float(np.mean(iou)), "per_image_dice": dice, "per_image_iou": iou,
            "predictions": predictions, "ms_per_image": 1000 * seconds / max(images, 1)}


def fit_segmentation(architecture, train, val, out_dir, epochs=15, batch_size=16, lr=1e-3, device=None, seed=0,
                     pretrained=True, log=print):
    """Train one model on masks only and keep the epoch with the best validation Dice.

    Returns (model with its best weights, history). The depth head exists but is not trained."""
    device = device or pick_device()
    torch.manual_seed(seed)
    model = build(architecture, train.size, pretrained).to(device)
    if hasattr(model, "compute_prototype"):
        used = model.compute_prototype(train, device)
        log(f"{architecture}: class prototype from {used} training images")
    optimiser = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs, eta_min=lr * 0.05)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=len(train) > batch_size,
                        generator=torch.Generator().manual_seed(seed))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path, best, history = out_dir / f"pets_{architecture}.pt", -1.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        started, losses = time.time(), []
        for batch in loader:
            logits, _ = model(batch["image"].to(device), batch["meta"].to(device))
            loss = mask_loss(logits, batch["labels"].to(device), 1)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        scheduler.step()
        scores = evaluate_segmentation(model, val, device, batch_size)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_dice": scores["dice"],
               "val_iou": scores["iou"], "seconds": time.time() - started}
        history.append(row)
        improved = row["val_dice"] > best
        if improved:
            best = row["val_dice"]
            torch.save(model.state_dict(), path)
        log(f"{architecture:11} epoch {epoch:3d}  loss {row['train_loss']:.4f}  val Dice {row['val_dice']:.4f}  "
            f"IoU {row['val_iou']:.4f}  ({row['seconds']:.0f}s){'  *' if improved else ''}")
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval(), history


def paired_difference(a, b, n_resamples=2000, seed=0):
    """Mean of (b - a) over the same test images, with a 95% bootstrap interval.

    If the interval excludes zero, the difference is unlikely to be chance."""
    difference = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    means = rng.choice(difference, size=(n_resamples, len(difference)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(difference.mean()), float(low), float(high)


def plot_curves(histories):
    """Training loss and validation Dice per epoch. `histories` maps a label to its history."""
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8))
    for (label, history), colour in zip(histories.items(), ("#2f6fb5", "#e07b22")):
        epochs = [row["epoch"] for row in history]
        left.plot(epochs, [row["train_loss"] for row in history], color=colour, label=label)
        right.plot(epochs, [row["val_dice"] for row in history], color=colour, label=label)
    for axis, title in ((left, "training loss"), (right, "validation Dice")):
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def plot_examples(dataset, results, n=6):
    """Photos where the two models differ most: the photo, the true mask, and each prediction."""
    import matplotlib.pyplot as plt

    labels = list(results)
    gap = np.asarray(results[labels[1]]["per_image_dice"]) - np.asarray(results[labels[0]]["per_image_dice"])
    order = np.argsort(gap)
    chosen = list(order[: n // 2]) + list(order[::-1][: n - n // 2])
    figure, axes = plt.subplots(len(chosen), 4, figsize=(11, 2.7 * len(chosen)), squeeze=False)
    for row, position in zip(axes, chosen):
        item = dataset[int(position)]
        truth = np.where(item["labels"].numpy() == IGNORE_LABEL, 0.5, item["labels"].numpy())
        row[0].imshow(item["image"].permute(1, 2, 0).numpy())
        row[1].imshow(truth, cmap="gray", vmin=0, vmax=1)
        row[0].set_title("photo", fontsize=9)
        row[1].set_title("true mask (grey = ignored border)", fontsize=9)
        for axis, label in zip(row[2:], labels):
            axis.imshow(results[label]["predictions"][int(position)], cmap="gray")
            axis.set_title(f"{label}: Dice {results[label]['per_image_dice'][int(position)]:.3f}", fontsize=9)
        for axis in row:
            axis.set_axis_off()
    figure.tight_layout()
    return figure
