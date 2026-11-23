"""Matplotlib figures for the notebooks. Every function returns the figure it draws."""

import numpy as np

from .masks import overlay

BLUE, ORANGE, GREY, RED = "#2f6fb5", "#e07b22", "#8a8a8a", "#c0392b"


def _plt():
    import matplotlib.pyplot as plt

    return plt


def calibration_fit(calibration, points):
    """Lens position against LiDAR distance with the fitted curve, and the residuals below."""
    plt = _plt()
    lens, distance = points["lens_position"], points["distance_m"] * 100
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True, height_ratios=[2, 1])
    grid = np.linspace(lens.min(), lens.max(), 300)
    top.scatter(lens, distance, s=14, color=GREY, alpha=0.7, label="captures (LiDAR at tap)")
    top.plot(grid, calibration.focus_distance_m(grid) * 100, color=BLUE, lw=2, label="calibration")
    top.set_ylabel("distance (cm)")
    top.set_title(f"Lens calibration · {calibration.n_samples} samples · RMSE {calibration.rmse_m * 100:.2f} cm")
    top.legend(frameon=False)
    residual = calibration.focus_distance_m(lens) * 100 - distance
    bottom.axhline(0, color=GREY, lw=1)
    bottom.scatter(lens, residual, s=14, color=BLUE, alpha=0.7)
    bottom.set_xlabel("lensPosition (0 = nearest, 1 = furthest)")
    bottom.set_ylabel("residual (cm)")
    for axis in (top, bottom):
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def learning_curves(histories):
    """Loss, Dice and object-distance error per epoch. `histories` maps a run name to its history."""
    plt = _plt()
    figure, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for (name, history), colour in zip(histories.items(), (BLUE, ORANGE, GREY, RED)):
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], color=colour, ls="--", label=f"{name} train")
        axes[0].plot(epochs, [row["val_loss"] for row in history], color=colour, label=f"{name} val")
        axes[1].plot(epochs, [row["val_dice"] for row in history], color=colour, label=name)
        axes[2].plot(epochs, [row["val_distance_error_cm"] for row in history], color=colour, label=name)
    for axis, title in zip(axes, ("loss", "validation foreground Dice", "validation object-distance error (cm)")):
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_yscale("log")
    axes[2].set_yscale("log")
    figure.tight_layout()
    return figure


def _columns(rows, *keys):
    return [np.array([row[key] for row in rows], dtype=float) for key in keys]


def predicted_vs_true_distance(results):
    """Predicted against true (LiDAR) distance. `results` maps a model name to its object rows."""
    plt = _plt()
    figure, axis = plt.subplots(figsize=(5.5, 5.5))
    limit = 0
    for (name, rows), colour in zip(results.items(), (BLUE, ORANGE, GREY, RED)):
        true, pred = _columns(rows, "true_distance_cm", "pred_distance_cm")
        axis.scatter(true, pred, s=16, alpha=0.7, color=colour, label=name)
        limit = max(limit, np.nanmax(true, initial=0), np.nanmax(pred, initial=0))
    axis.plot([0, limit], [0, limit], color=GREY, lw=1)
    axis.set_xlabel("LiDAR distance (cm)")
    axis.set_ylabel("predicted distance (cm)")
    axis.set_aspect("equal")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def error_vs_distance(results, key="pred_distance_cm", truth="true_distance_cm", ylabel="distance error (cm)"):
    """Signed error against true distance, one colour per model."""
    plt = _plt()
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.axhline(0, color=GREY, lw=1)
    for (name, rows), colour in zip(results.items(), (BLUE, ORANGE, GREY, RED)):
        distance, pred, true = _columns(rows, "true_distance_cm", key, truth)
        axis.scatter(distance, pred - true, s=16, alpha=0.7, color=colour, label=name)
    axis.set_xlabel("LiDAR distance (cm)")
    axis.set_ylabel(ylabel)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def error_vs_lens(rows):
    """Distance error against the raw lens reading; a trend here suggests calibration drift."""
    plt = _plt()
    lens, pred, true = _columns(rows, "lens_position", "pred_distance_cm", "true_distance_cm")
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.axhline(0, color=GREY, lw=1)
    axis.scatter(lens, pred - true, s=16, alpha=0.7, color=BLUE)
    axis.set_xlabel("lensPosition")
    axis.set_ylabel("distance error (cm)")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def example_overlays(dataset, predictions, object_rows, n=4):
    """Best and worst samples by object Dice: predicted mask overlay with predicted and LiDAR numbers."""
    plt = _plt()
    by_index = {}
    for row in object_rows:
        by_index.setdefault(row["index"], []).append(row)
    ranked = sorted(by_index, key=lambda i: np.mean([row["dice"] for row in by_index[i]]))
    chosen = [("worst", i) for i in ranked[:n]] + [("best", i) for i in ranked[::-1][:n]]
    figure, axes = plt.subplots(2, n, figsize=(4.2 * n, 6.6), squeeze=False)
    for axis in axes.ravel():
        axis.set_axis_off()
    for (kind, index), axis in zip(chosen, axes.ravel()):
        item = dataset[index]
        image = (item["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pred_labels, _ = predictions[index]
        axis.imshow(overlay(image, [(pred_labels > 0).numpy()]))
        axis.contour((item["instances"] > 0).numpy(), levels=[0.5], colors="white", linewidths=0.8)
        lines = [f"{kind} · {item['sample_id']}"]
        for row in by_index[index]:
            lines.append(f"Dice {row['dice']:.2f} · {row['pred_distance_cm']:.1f} cm (LiDAR {row['true_distance_cm']:.1f}) · "
                         f"{row['pred_width_cm']:.1f}×{row['pred_height_cm']:.1f} cm")
        axis.set_title("\n".join(lines), fontsize=8)
    figure.suptitle("Predicted mask (tint) against reviewed mask (white outline)", fontsize=10)
    figure.tight_layout()
    return figure
