"""Mask, distance and size metrics, and the evaluation loop (used by 03_evaluate.ipynb).

The model is given the image and the meta vector only. LiDAR depth appears here solely as the
ground truth the predictions are scored against.
"""

import numpy as np
import torch

from .geometry import object_size_m, object_size_robust_m
from .session import IGNORE_LABEL
from .train import foreground

MIN_OBJECT_PIXELS = 30


def dice_iou(pred, true):
    """Dice and IoU of two boolean tensors (both 1.0 when both are empty)."""
    intersection = float((pred & true).sum())
    total = float(pred.sum() + true.sum())
    if total == 0:
        return 1.0, 1.0
    return 2 * intersection / total, intersection / (total - intersection)


def components(mask):
    """Connected components of a boolean [H, W] tensor, as a list of boolean tensors."""
    from scipy import ndimage

    labelled, n = ndimage.label(mask.cpu().numpy())
    return [torch.from_numpy(labelled == i) for i in range(1, n + 1)]


def _sizes(mask, depth, k):
    """Size by the reference method and by the robust method, in cm: (w1, h1, w2, h2, distance)."""
    w1, h1, d = object_size_m(mask, depth, *k)
    w2, h2, _ = object_size_robust_m(mask, depth, *k)
    return w1 * 100, h1 * 100, w2 * 100, h2 * 100, d * 100


def evaluate_sample(pred_labels, pred_depth, item, sample_meta=None):
    """Score one sample. All tensors are [H, W] on the CPU at the model's resolution.

    Returns (object_rows, image_row). Each true object is matched to the predicted connected
    component of its class that overlaps it most; an object with no overlapping component is
    reported as missed, with nan predictions."""
    k = [float(x) for x in item["intrinsics"]]
    labels, instances = item["labels"], item["instances"]
    lidar, valid = item["depth"], item["depth_valid"]
    true_fg = (labels > 0) & (labels != IGNORE_LABEL)
    pred_fg = (pred_labels > 0) & (labels != IGNORE_LABEL)
    image_dice, image_iou = dice_iou(pred_fg, true_fg)
    n_objects = int(instances.max())
    image_row = {
        "session_id": item["session_id"], "sample_id": item["sample_id"], "background_only": n_objects == 0,
        "dice": image_dice, "iou": image_iou, "false_positive_fraction": float((pred_fg & ~true_fg).float().mean()),
        "predicted_fraction": float(pred_fg.float().mean()),
    }
    rows = []
    blobs_by_class = {}
    for index in range(1, n_objects + 1):
        true_mask = instances == index
        if int(true_mask.sum()) < MIN_OBJECT_PIXELS:
            continue
        class_id = int(labels[true_mask].mode().values)
        if class_id not in blobs_by_class:
            blobs_by_class[class_id] = components(pred_labels == class_id)
        overlaps = [int((blob & true_mask).sum()) for blob in blobs_by_class[class_id]]
        matched = blobs_by_class[class_id][int(np.argmax(overlaps))] if overlaps and max(overlaps) > 0 else None

        lidar_region = true_mask & valid
        lidar_for_size = torch.where(valid, lidar, lidar[lidar_region].median() if lidar_region.any() else lidar)
        _, _, true_w, true_h, true_d = _sizes(true_mask, lidar_for_size, k)
        true_d = float(lidar[lidar_region].median()) * 100 if lidar_region.any() else float("nan")
        row = {
            "session_id": item["session_id"], "sample_id": item["sample_id"], "object": index, "class_id": class_id,
            "is_focus_target": index - 1 == item["focus_tap_index"], "lens_position": item["lens_position"],
            "focus_distance_cm": item["focus_distance_m"] * 100, "area_px": int(true_mask.sum()),
            "lidar_px": int(lidar_region.sum()), "missed": matched is None,
            "true_distance_cm": true_d, "true_width_cm": true_w, "true_height_cm": true_h,
            "measured_width_cm": float("nan"), "measured_height_cm": float("nan"), "measured_distance_cm": float("nan"),
        }
        if sample_meta is not None:
            obj = sample_meta["objects"][index - 1]
            size = obj.get("true_size_cm") or {}
            row["measured_width_cm"] = _number(size.get("width"))
            row["measured_height_cm"] = _number(size.get("height"))
            row["measured_distance_cm"] = _number(obj.get("true_distance_cm"))
        if matched is None:
            row.update(dice=0.0, iou=0.0, **{key: float("nan") for key in (
                "pred_distance_cm", "pred_width_cm", "pred_height_cm", "pred_width_raw_cm", "pred_height_raw_cm")})
        else:
            d, i = dice_iou(matched, true_mask)
            w1, h1, w2, h2, pred_d = _sizes(matched, pred_depth, k)
            row.update(dice=d, iou=i, pred_distance_cm=pred_d, pred_width_cm=w2, pred_height_cm=h2,
                       pred_width_raw_cm=w1, pred_height_raw_cm=h1)
        rows.append(row)
    return rows, image_row


def _number(value):
    return float("nan") if value is None else float(value)


@torch.no_grad()
def evaluate_model(model, dataset, device="cpu", batch_size=8):
    """Run the model over a dataset (image + meta only) and score every sample.

    Returns (object_rows, image_rows, predictions) where predictions maps dataset index to
    (pred_labels, pred_depth) CPU tensors for plotting."""
    from torch.utils.data import DataLoader

    model = model.to(device).eval()
    object_rows, image_rows, predictions = [], [], {}
    for batch in DataLoader(dataset, batch_size=batch_size):
        mask_logits, pred_depth = model(batch["image"].to(device), batch["meta"].to(device))
        pred_labels = foreground(mask_logits, model.n_classes).cpu()
        pred_depth = pred_depth[:, 0].cpu()
        for b in range(len(batch["image"])):
            index = int(batch["index"][b])
            item = {key: (value[b] if torch.is_tensor(value) else value[b]) for key, value in batch.items()}
            for key in ("focus_distance_m", "lens_position", "focus_tap_index"):
                item[key] = float(item[key]) if key != "focus_tap_index" else int(item[key])
            rows, image_row = evaluate_sample(pred_labels[b], pred_depth[b], item, dataset.sample_meta(index))
            for row in rows:
                row["index"] = index
            image_row["index"] = index
            object_rows.extend(rows)
            image_rows.append(image_row)
            predictions[index] = (pred_labels[b], pred_depth[b])
    return object_rows, image_rows, predictions


def _column(rows, key):
    return np.array([row[key] for row in rows], dtype=float)


def _error_stats(pred, true):
    """Mean and median absolute error and mean absolute percentage error, ignoring nans."""
    ok = np.isfinite(pred) & np.isfinite(true) & (true > 0)
    if not ok.any():
        return {"n": 0, "mae": float("nan"), "median_ae": float("nan"), "mape": float("nan"), "bias": float("nan")}
    error = pred[ok] - true[ok]
    return {"n": int(ok.sum()), "mae": float(np.abs(error).mean()), "median_ae": float(np.median(np.abs(error))),
            "mape": float((np.abs(error) / true[ok]).mean() * 100), "bias": float(error.mean())}


def summarise(object_rows, image_rows):
    """Headline numbers for one model: masks, distance, size (raw and bias-fixed), false positives."""
    summary = {"n_objects": len(object_rows), "n_images": len(image_rows)}
    if object_rows:
        summary["missed_fraction"] = float(np.mean([row["missed"] for row in object_rows]))
        summary["dice"] = float(_column(object_rows, "dice").mean())
        summary["iou"] = float(_column(object_rows, "iou").mean())
        true_d = _column(object_rows, "true_distance_cm")
        summary["distance"] = _error_stats(_column(object_rows, "pred_distance_cm"), true_d)
        focused = np.array([row["is_focus_target"] for row in object_rows])
        focus = np.where(focused, _column(object_rows, "focus_distance_cm"), np.nan)
        summary["distance_focus_baseline"] = _error_stats(focus, true_d)
        summary["distance_model_on_focus_targets"] = _error_stats(
            np.where(focused, _column(object_rows, "pred_distance_cm"), np.nan), true_d)
        for axis in ("width", "height"):
            true_size = _column(object_rows, f"true_{axis}_cm")
            measured = _column(object_rows, f"measured_{axis}_cm")
            summary[f"{axis}_vs_lidar"] = _error_stats(_column(object_rows, f"pred_{axis}_cm"), true_size)
            summary[f"{axis}_vs_lidar_raw"] = _error_stats(_column(object_rows, f"pred_{axis}_raw_cm"), true_size)
            summary[f"{axis}_vs_measured"] = _error_stats(_column(object_rows, f"pred_{axis}_cm"), measured)
            summary[f"{axis}_vs_measured_raw"] = _error_stats(_column(object_rows, f"pred_{axis}_raw_cm"), measured)
            summary[f"lidar_{axis}_vs_measured"] = _error_stats(true_size, measured)
    background = [row for row in image_rows if row["background_only"]]
    summary["n_background_only"] = len(background)
    summary["background_false_positive_fraction"] = (
        float(np.mean([row["predicted_fraction"] for row in background])) if background else float("nan"))
    return summary


def correlation(object_rows, x_key, y_key):
    """Pearson correlation between two per-object columns, ignoring nans (nan if undefined)."""
    x, y = _column(object_rows, x_key), _column(object_rows, y_key)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or x[ok].std() == 0 or y[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def distance_error_by_bin(object_rows, bins_cm=(15, 25, 35, 45, 55, 70, 100), key="pred_distance_cm"):
    """Distance error grouped by true distance. Returns a list of dicts, one per non-empty bin."""
    true_d, pred = _column(object_rows, "true_distance_cm"), _column(object_rows, key)
    table = []
    for lo, hi in zip(bins_cm[:-1], bins_cm[1:]):
        inside = (true_d >= lo) & (true_d < hi)
        stats = _error_stats(pred[inside], true_d[inside])
        if stats["n"]:
            table.append({"bin_cm": (lo, hi), **stats})
    return table


def format_summary(summaries):
    """Side-by-side text table of summarise() results. `summaries` maps a model name to its summary."""
    def stat(summary, key, field="mae", unit="cm"):
        value = summary.get(key, {}).get(field, float("nan"))
        return "    n/a" if not np.isfinite(value) else f"{value:6.2f} {unit}"

    rows = [
        ("objects", lambda s: f"{s['n_objects']:9d}"),
        ("missed objects", lambda s: f"{s.get('missed_fraction', float('nan')):8.1%} "),
        ("mask Dice", lambda s: f"{s.get('dice', float('nan')):9.3f}"),
        ("mask IoU", lambda s: f"{s.get('iou', float('nan')):9.3f}"),
        ("distance error (mean abs)", lambda s: stat(s, "distance")),
        ("distance error (median abs)", lambda s: stat(s, "distance", "median_ae")),
        ("distance error (%)", lambda s: stat(s, "distance", "mape", "% ")),
        ("width error vs LiDAR size", lambda s: stat(s, "width_vs_lidar")),
        ("width error vs LiDAR size (%)", lambda s: stat(s, "width_vs_lidar", "mape", "% ")),
        ("height error vs LiDAR size", lambda s: stat(s, "height_vs_lidar")),
        ("height error vs LiDAR size (%)", lambda s: stat(s, "height_vs_lidar", "mape", "% ")),
        ("width error vs ruler", lambda s: stat(s, "width_vs_measured")),
        ("height error vs ruler", lambda s: stat(s, "height_vs_measured")),
        ("false-positive area, background-only", lambda s: f"{s['background_false_positive_fraction']:8.3%} "
         if np.isfinite(s["background_false_positive_fraction"]) else "      n/a"),
    ]
    names = list(summaries)
    lines = [f"{'':38}" + "".join(f"{name:>18}" for name in names)]
    for label, cell in rows:
        lines.append(f"{label:38}" + "".join(f"{cell(summaries[name]):>18}" for name in names))
    return "\n".join(lines)


def format_size_fix(summary):
    """Size error before and after the bias fix (reference percentile method vs robust method)."""
    lines = [f"{'':24}{'reference method':>20}{'after fix':>14}"]
    for target, label in (("lidar", "vs LiDAR size"), ("measured", "vs ruler")):
        for axis in ("width", "height"):
            raw, fixed = summary[f"{axis}_vs_{target}_raw"], summary[f"{axis}_vs_{target}"]
            if fixed["n"]:
                lines.append(f"{axis + ' ' + label:24}{raw['mae']:13.2f} cm ({raw['bias']:+.2f}){fixed['mae']:8.2f} cm ({fixed['bias']:+.2f})")
    lines.append("(mean absolute error, with the mean signed error in brackets)")
    return "\n".join(lines)


def verdict(full, image_only, object_rows, good_percent=10.0):
    """A short plain-language reading of the evaluation, generated from the numbers."""
    lines = []
    d_full, d_ablation = full["distance"], image_only["distance"]
    baseline, on_targets = full["distance_focus_baseline"], full["distance_model_on_focus_targets"]
    if d_full["n"] == 0:
        return "No objects were detected, so there is nothing to judge yet."
    if full.get("missed_fraction", 0) > 0.5:
        lines.append(f"Caution: {full['missed_fraction']:.0%} of objects were missed, so the numbers below rest on "
                     f"only {d_full['n']} detected object(s) and say little. Train longer or check the masks.")
    if np.isfinite(d_ablation["mae"]) and d_ablation["mae"] > 0:
        gain = 1 - d_full["mae"] / d_ablation["mae"]
        if gain > 0.1:
            helps = f"helps: distance error is {gain:.0%} lower with it"
        elif gain < -0.1:
            helps = f"does not help: distance error is {-gain:.0%} higher with it"
        else:
            helps = "makes no measurable difference to distance error"
        lines.append(f"Focus input {helps} ({d_full['mae']:.2f} cm with focus, {d_ablation['mae']:.2f} cm image-only, "
                     f"mean absolute error over {d_full['n']} objects).")
    if baseline["n"]:
        lines.append(f"On the focused objects the model's distance error is {on_targets['mae']:.2f} cm; using the "
                     f"calibrated focus distance alone, with no model, gives {baseline['mae']:.2f} cm.")
    good = [row for row in distance_error_by_bin(object_rows) if row["n"] and row["mape"] <= good_percent]
    if good:
        lines.append(f"Distance error stays within {good_percent:.0f}% in these ranges: "
                     + ", ".join(f"{lo}-{hi} cm ({row['mape']:.1f}%, n={row['n']})"
                                 for row in good for lo, hi in [row["bin_cm"]]) + ".")
    else:
        lines.append(f"No distance range reaches {good_percent:.0f}% distance error.")
    drift = correlation([dict(row, error=row["pred_distance_cm"] - row["true_distance_cm"]) for row in object_rows],
                        "lens_position", "error")
    if np.isfinite(drift):
        lines.append(f"Correlation between the lens reading and the distance error: {drift:+.2f} "
                     + ("(a trend this strong suggests the lens calibration has drifted)." if abs(drift) > 0.5
                        else "(no strong sign of calibration drift)."))
    if 0 < full.get("missed_fraction", 0) <= 0.5:
        lines.append(f"{full['missed_fraction']:.0%} of objects were missed entirely.")
    return "\n".join(lines)
