"""Phase 1 checks on a real capture: depth/RGB alignment overlays and LiDAR coverage by distance.

    python -m unetultra.capture_report path/to/session.zip --out data/report

Writes one overlay PNG per sample (depth edges drawn on the RGB image) and prints how many valid,
high-confidence LiDAR pixels land on the object at each distance, with a suggested safe minimum.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .geometry import CenterSquareCrop, Intrinsics, nearest_resize
from .session import Session, image_to_depth_point, lidar_at_point

DEFAULT_BINS_CM = (10, 15, 20, 25, 30, 40, 50, 60, 70, 85, 100)
GOOD_COVERAGE = 0.7
DEPTH_BAND_M = 0.03


def object_coverage(sess, sample_id, object_size_m=0.055):
    """For each object in a sample: LiDAR distance at the tap and how well depth covers the object.

    Coverage is the share of depth pixels in the object's expected footprint (its reviewed mask
    if one exists, otherwise a box of the nominal object size around the tap) that hold a valid
    reading within 3 cm of the tap distance."""
    meta = sess.meta(sample_id)
    depth, valid = sess.depth_m(sample_id), sess.depth_valid(sample_id)
    k_depth = Intrinsics.from_dict(meta["intrinsics_depth"])
    image_size, depth_size = tuple(meta["image_size"]), tuple(meta["depth_size"])
    instances = sess.instances(sample_id)
    rows = []
    for index, obj in enumerate(meta["objects"], start=1):
        distance, n_tap = lidar_at_point(depth, valid, obj["tap_point_image"], image_size)
        row = {
            "sample_id": sample_id, "object": index, "distance_m": distance, "n_valid_at_tap": n_tap,
            "footprint_px": 0, "valid_px": 0, "coverage": float("nan"), "lens_position": meta["lens_position"],
        }
        if np.isfinite(distance):
            if instances is not None and (instances == index).any():
                footprint = nearest_resize(instances, depth_size) == index
            else:
                u, v = image_to_depth_point(obj["tap_point_image"], image_size, depth_size)
                half_u = k_depth.fx * object_size_m / (2 * distance)
                half_v = k_depth.fy * object_size_m / (2 * distance)
                uu, vv = np.meshgrid(np.arange(depth_size[0]), np.arange(depth_size[1]))
                footprint = (np.abs(uu - u) <= half_u) & (np.abs(vv - v) <= half_v)
            on_object = footprint & valid & (np.abs(depth - distance) < DEPTH_BAND_M)
            row["footprint_px"] = int(footprint.sum())
            row["valid_px"] = int(on_object.sum())
            row["coverage"] = row["valid_px"] / max(row["footprint_px"], 1)
        rows.append(row)
    return rows


def coverage_by_distance(rows, bins_cm=DEFAULT_BINS_CM):
    """Group object rows into distance bins. Returns one dict per non-empty bin."""
    table = []
    for lo, hi in zip(bins_cm[:-1], bins_cm[1:]):
        group = [r for r in rows if np.isfinite(r["distance_m"]) and lo <= r["distance_m"] * 100 < hi]
        if group:
            table.append({
                "bin_cm": (lo, hi), "n_objects": len(group),
                "median_valid_px": float(np.median([r["valid_px"] for r in group])),
                "median_coverage": float(np.median([r["coverage"] for r in group])),
            })
    return table


def safe_minimum_cm(table):
    """Lower edge of the nearest bin from which every further bin has good coverage, or None."""
    safe = None
    for row in reversed(table):
        if row["median_coverage"] < GOOD_COVERAGE:
            break
        safe = row["bin_cm"][0]
    return safe


def format_report(rows, table):
    no_reading = sum(1 for r in rows if not np.isfinite(r["distance_m"]))
    lines = ["distance      objects   LiDAR px on object (median)   coverage (median)"]
    for row in table:
        lines.append(
            f"{row['bin_cm'][0]:>3}-{row['bin_cm'][1]:<3} cm   {row['n_objects']:>5}     "
            f"{row['median_valid_px']:>10.0f}                  {row['median_coverage']:>6.0%}"
        )
    if no_reading:
        lines.append(f"{no_reading} object(s) had no valid LiDAR reading at the tap at all.")
    safe = safe_minimum_cm(table)
    if not table:
        lines.append("No objects with a LiDAR reading; nothing to report.")
    elif safe is None:
        lines.append(f"No distance bin reaches {GOOD_COVERAGE:.0%} coverage: LiDAR labels look unreliable here.")
    else:
        lines.append(f"Safe minimum distance on this data: about {safe} cm "
                     f"(every bin from there outwards has at least {GOOD_COVERAGE:.0%} coverage).")
    return "\n".join(lines)


def depth_edge_overlay(sess, sample_id, edge_threshold_m=0.03):
    """RGB image with depth discontinuities in red, missing depth tinted blue, taps in yellow, and
    the centre square the model sees outlined in white with the rest dimmed.

    If image and depth are aligned (same orientation, valid intrinsics), the red lines trace the
    object outlines in the photo. Reviewed masks, when present, are outlined in green."""
    meta = sess.meta(sample_id)
    image = sess.image(sample_id).copy()
    size = tuple(meta["image_size"])
    depth = nearest_resize(sess.depth_m(sample_id), size)
    valid = nearest_resize(sess.depth_valid(sample_id), size)
    scale = max(1, size[0] // meta["depth_size"][0])
    edges = np.zeros(depth.shape, dtype=bool)
    edges[:, scale:] |= np.abs(depth[:, scale:] - depth[:, :-scale]) > edge_threshold_m
    edges[scale:, :] |= np.abs(depth[scale:, :] - depth[:-scale, :]) > edge_threshold_m
    edges &= valid
    image[~valid] = (0.6 * image[~valid] + 0.4 * np.array([40, 80, 255])).astype(np.uint8)
    image[edges] = (255, 0, 0)
    instances = sess.instances(sample_id)
    if instances is not None:
        outline = np.zeros(depth.shape, dtype=bool)
        outline[:, 1:] |= instances[:, 1:] != instances[:, :-1]
        outline[1:, :] |= instances[1:, :] != instances[:-1, :]
        image[outline] = (0, 255, 0)
    crop = CenterSquareCrop.for_size(size)
    outside = np.ones(depth.shape, dtype=bool)
    outside[crop.y0 : crop.y0 + crop.side, crop.x0 : crop.x0 + crop.side] = False
    image[outside] = image[outside] // 2
    picture = Image.fromarray(image)
    draw = ImageDraw.Draw(picture)
    draw.rectangle([crop.x0, crop.y0, crop.x0 + crop.side - 1, crop.y0 + crop.side - 1], outline=(255, 255, 255), width=2)
    radius = max(4, size[0] // 160)
    for obj in meta["objects"]:
        x, y = obj["tap_point_image"]
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=(255, 230, 0), width=3)
    return np.asarray(picture)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="session zip(s)")
    parser.add_argument("--out", help="directory for overlay PNGs (omit to skip overlays)")
    parser.add_argument("--object-size-cm", type=float, default=5.5, help="nominal object size")
    args = parser.parse_args(argv)
    rows = []
    for path in args.paths:
        with Session(path) as sess:
            for sample_id in sess.sample_ids:
                rows.extend(object_coverage(sess, sample_id, args.object_size_cm / 100))
                if args.out:
                    out = Path(args.out) / sess.session_id
                    out.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(depth_edge_overlay(sess, sample_id)).save(out / f"{sample_id}.jpg", quality=90)
    print(format_report(rows, coverage_by_distance(rows)))
    if args.out:
        print(f"Overlays written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
