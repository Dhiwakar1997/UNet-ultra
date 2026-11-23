"""Generate a synthetic capture session in exactly the real data format.

Flat shapes of known size sit fronto-parallel at 15-70 cm in front of a textured back wall.
Image, depth and intrinsics are mutually consistent, so every notebook can be built and
tested before any real captures exist. Generation is deterministic for a given seed.
"""

import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from . import session as fmt
from .geometry import Intrinsics, nearest_resize

# Roughly an iPhone 1x wide camera at 1920x1440, with an off-centre principal point so that
# code which wrongly assumes the image centre fails the tests.
REFERENCE_SIZE = (1920, 1440)
REFERENCE_INTRINSICS = Intrinsics(fx=1450.0, fy=1450.0, cx=972.0, cy=708.0)
DEPTH_SIZE = (256, 192)
MIN_DISTANCE_M, MAX_DISTANCE_M = 0.15, 0.70
OBJECT_COLOURS = ((214, 64, 52), (52, 120, 214), (240, 190, 40), (60, 170, 90))


def fake_lens_position(distance_m):
    """The synthetic device's lens curve: monotonic, non-linear, and 0..1 like the real reading.

    As on iOS, 0 is the nearest focus and 1 the furthest."""
    return float(np.clip(1.0 - 0.125 / distance_m, 0.0, 1.0))


def _png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _place_object(rng, k, size, taken):
    """Pick a size, distance and position for one object that fits in the image and avoids `taken`.

    Returns a dict with the metric truth and the pixel-space half-extents, or None if no room."""
    width, height = size
    for _ in range(50):
        w_m = rng.uniform(0.045, 0.065)
        h_m = rng.uniform(0.045, 0.065)
        z = rng.uniform(MIN_DISTANCE_M, MAX_DISTANCE_M)
        half_u, half_v = k.fx * w_m / (2 * z), k.fy * h_m / (2 * z)
        margin = 6
        if 2 * (half_u + margin) >= width or 2 * (half_v + margin) >= height:
            continue
        u0 = rng.uniform(half_u + margin, width - 1 - half_u - margin)
        v0 = rng.uniform(half_v + margin, height - 1 - half_v - margin)
        box = (u0 - half_u - margin, v0 - half_v - margin, u0 + half_u + margin, v0 + half_v + margin)
        if any(box[0] < t[2] and t[0] < box[2] and box[1] < t[3] and t[1] < box[3] for t in taken):
            continue
        taken.append(box)
        return {
            "w_m": w_m, "h_m": h_m, "z": z, "u0": u0, "v0": v0, "half_u": half_u, "half_v": half_v,
            "ellipse": bool(rng.random() < 0.5),
        }
    return None


def _render_sample(rng, k, size, depth_size, n_objects, n_classes):
    """Render one sample. Returns image, depth_mm, confidence, labels, instances and object dicts."""
    width, height = size
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

    # Back wall: low-frequency colour texture, 0.9-1.6 m away and slightly tilted.
    coarse = rng.uniform(70, 190, size=(8, 10, 3)).astype(np.uint8)
    image = np.asarray(Image.fromarray(coarse).resize(size, Image.BICUBIC), dtype=np.float32)
    wall_z = rng.uniform(0.9, 1.6)
    depth = wall_z * (1 + rng.uniform(-0.1, 0.1) * (uu / width - 0.5) + rng.uniform(-0.1, 0.1) * (vv / height - 0.5))
    labels = np.zeros((height, width), dtype=np.uint8)
    instances = np.zeros((height, width), dtype=np.uint8)

    placed, taken = [], []
    for _ in range(n_objects):
        obj = _place_object(rng, k, size, taken)
        if obj is not None:
            placed.append(obj)
    # Paint far-to-near; boxes never overlap, so the order only matters for consistency.
    objects = []
    for index, obj in enumerate(placed, start=1):
        du, dv = (uu - obj["u0"]) / obj["half_u"], (vv - obj["v0"]) / obj["half_v"]
        inside = (du**2 + dv**2 <= 1) if obj["ellipse"] else ((np.abs(du) <= 1) & (np.abs(dv) <= 1))
        class_id = int(rng.integers(1, n_classes))
        colour = np.array(OBJECT_COLOURS[(class_id - 1) % len(OBJECT_COLOURS)], dtype=np.float32)
        image[inside] = colour * rng.uniform(0.85, 1.1)
        depth[inside] = obj["z"]
        labels[inside] = class_id
        instances[inside] = index
        obj["class_id"] = class_id
        objects.append(obj)

    image = np.clip(image + rng.normal(0, 3, image.shape), 0, 255).astype(np.uint8)
    depth_small = nearest_resize(depth, depth_size)
    depth_small = depth_small + rng.normal(0, 0.0005, depth_small.shape)
    depth_mm = np.clip(np.round(depth_small * 1000), 0, 65535).astype(np.uint16)
    confidence = np.full(depth_mm.shape, fmt.HIGH_CONFIDENCE, dtype=np.uint8)
    dropouts = rng.random(depth_mm.shape) < 0.01
    confidence[dropouts] = rng.integers(0, 2, size=int(dropouts.sum()))
    depth_mm[dropouts & (confidence == 0)] = 0
    return image, depth_mm, confidence, labels, instances, objects, wall_z


def make_fake_session(
    out_path,
    n_samples=24,
    split="train",
    classes=("background", "airpods_case"),
    image_size=(960, 720),
    depth_size=DEPTH_SIZE,
    seed=0,
    with_masks=False,
    session_id=None,
):
    """Write a synthetic session zip to out_path and return the path.

    About 15% of samples are background-only and about 25% hold two objects. With
    with_masks=True the true labels.png / instances.png are included and marked accepted;
    otherwise masks are "pending", as they are when a zip leaves the phone.
    """
    rng = np.random.default_rng(seed)
    classes = list(classes)
    session_id = session_id or f"fake{seed:04d}"
    k_image = REFERENCE_INTRINSICS.scaled(REFERENCE_SIZE, image_size)
    k_depth = k_image.scaled(image_size, depth_size)
    start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Guarantee at least one background-only and one two-object sample in any session of 3+.
    counts = rng.choice([0, 1, 2], size=n_samples, p=[0.15, 0.60, 0.25])
    if n_samples >= 3:
        counts[:3] = [1, 2, 0]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_ids, mask_status = [], {}
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as zf:
        for i, n_objects in enumerate(counts):
            sample_id = f"{i:04d}"
            image, depth_mm, confidence, labels, instances, objects, wall_z = _render_sample(
                rng, k_image, image_size, depth_size, int(n_objects), len(classes)
            )
            focus_distance = objects[0]["z"] if objects else wall_z
            lens = float(np.clip(fake_lens_position(focus_distance) + rng.normal(0, 0.004), 0, 1))
            meta = {
                "sample_id": sample_id,
                "timestamp": (start + timedelta(seconds=5 * i)).isoformat(),
                "image_size": list(image_size),
                "depth_size": list(depth_size),
                "intrinsics_image": k_image.to_dict(),
                "intrinsics_depth": k_depth.to_dict(),
                "intrinsics_reference_size": list(REFERENCE_SIZE),
                "lens_position": lens,
                "focus_settled": True,
                "focus_tap_index": 0,
                "device_orientation": "landscapeRight",
                "exposure": {"iso": 100, "duration_s": 1 / 60},
                "background_only": not objects,
                "objects": [
                    {
                        "class_id": obj["class_id"],
                        "label": classes[obj["class_id"]],
                        "tap_point_image": [round(obj["u0"], 1), round(obj["v0"], 1)],
                        "extra_points": [],
                        "lidar_distance_at_tap_m": round(obj["z"], 4),
                        "true_size_cm": {"width": round(obj["w_m"] * 100, 2), "height": round(obj["h_m"] * 100, 2)},
                        "facing": "front face to camera",
                        "true_distance_cm": round(obj["z"] * 100, 2),
                    }
                    for obj in objects
                ],
                "notes": "synthetic",
            }
            prefix = f"samples/{sample_id}/"
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="JPEG", quality=95)
            zf.writestr(prefix + fmt.IMAGE_FILE, buffer.getvalue())
            zf.writestr(prefix + fmt.DEPTH_FILE, _png_bytes(depth_mm))
            zf.writestr(prefix + fmt.CONFIDENCE_FILE, _png_bytes(confidence))
            zf.writestr(prefix + fmt.META_FILE, json.dumps(meta, indent=2))
            if with_masks:
                zf.writestr(prefix + fmt.LABELS_FILE, _png_bytes(labels))
                zf.writestr(prefix + fmt.INSTANCES_FILE, _png_bytes(instances))
            sample_ids.append(sample_id)
            mask_status[sample_id] = "accepted" if with_masks else "pending"

        manifest = {
            "format_version": fmt.FORMAT_VERSION,
            "session_id": session_id,
            "split": split,
            "classes": classes,
            "device_model": "FakePhone1,1",
            "ios_version": "0.0",
            "app_version": "fake",
            "created_at": start.isoformat(),
            "sample_ids": sample_ids,
            "mask_status": mask_status,
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    return out_path
