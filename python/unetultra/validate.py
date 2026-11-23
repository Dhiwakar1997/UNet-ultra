"""Check that a session zip follows the data contract.

    python -m unetultra.validate path/to/session.zip [more.zip ...]
"""

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np

from . import session as fmt
from .geometry import Intrinsics
from .session import Session

MANIFEST_KEYS = (
    "format_version", "session_id", "split", "classes", "device_model", "ios_version",
    "app_version", "created_at", "sample_ids", "mask_status",
)
META_KEYS = (
    "sample_id", "timestamp", "image_size", "depth_size", "intrinsics_image", "intrinsics_depth",
    "intrinsics_reference_size", "lens_position", "focus_settled", "focus_tap_index",
    "device_orientation", "exposure", "background_only", "objects", "notes",
)
OBJECT_KEYS = ("class_id", "label", "tap_point_image")
# Depth intrinsics should be the image intrinsics rescaled; allow a little slack for rounding.
INTRINSICS_REL_TOL = 0.02
PRINCIPAL_POINT_TOL_PX = 2.0


@dataclass
class Report:
    """Validation result: errors per sample, plus session-level errors under the key None."""

    path: str
    n_samples: int = 0
    n_background_only: int = 0
    n_objects: int = 0
    errors: dict = field(default_factory=dict)

    def add(self, sample_id, message):
        self.errors.setdefault(sample_id, []).append(message)

    @property
    def ok(self):
        return not self.errors

    @property
    def n_valid_samples(self):
        return self.n_samples - sum(1 for k in self.errors if k is not None)

    def format(self):
        lines = [f"{self.path}"]
        for message in self.errors.get(None, []):
            lines.append(f"  session: {message}")
        for sample_id, messages in self.errors.items():
            if sample_id is not None:
                lines.extend(f"  {sample_id}: {m}" for m in messages)
        lines.append(
            f"  {self.n_valid_samples}/{self.n_samples} samples valid "
            f"({self.n_objects} objects, {self.n_background_only} background-only)"
        )
        lines.append("  OK" if self.ok else "  INVALID")
        return "\n".join(lines)


def _check_intrinsics(report, sample_id, meta):
    image_size, depth_size = meta["image_size"], meta["depth_size"]
    try:
        k_image = Intrinsics.from_dict(meta["intrinsics_image"])
        k_depth = Intrinsics.from_dict(meta["intrinsics_depth"])
    except (KeyError, TypeError, ValueError):
        report.add(sample_id, "intrinsics must hold numeric fx, fy, cx, cy")
        return
    if min(k_image.fx, k_image.fy, k_depth.fx, k_depth.fy) <= 0:
        report.add(sample_id, "focal lengths must be positive")
        return
    if not (0 <= k_image.cx < image_size[0] and 0 <= k_image.cy < image_size[1]):
        report.add(sample_id, "image principal point lies outside the image")
    expected = k_image.scaled(image_size, depth_size)
    if (
        abs(k_depth.fx / expected.fx - 1) > INTRINSICS_REL_TOL
        or abs(k_depth.fy / expected.fy - 1) > INTRINSICS_REL_TOL
        or abs(k_depth.cx - expected.cx) > PRINCIPAL_POINT_TOL_PX
        or abs(k_depth.cy - expected.cy) > PRINCIPAL_POINT_TOL_PX
    ):
        report.add(sample_id, "depth intrinsics are not the image intrinsics scaled to the depth size")


def _check_objects(report, sample_id, meta, n_classes):
    objects = meta["objects"]
    if meta["background_only"] != (len(objects) == 0):
        report.add(sample_id, "background_only must be true exactly when objects is empty")
    if objects and not 0 <= meta["focus_tap_index"] < len(objects):
        report.add(sample_id, "focus_tap_index does not point at an object")
    width, height = meta["image_size"]
    for i, obj in enumerate(objects):
        missing = [k for k in OBJECT_KEYS if k not in obj]
        if missing:
            report.add(sample_id, f"object {i} is missing {missing}")
            continue
        if not 1 <= obj["class_id"] < n_classes:
            report.add(sample_id, f"object {i} has class_id {obj['class_id']} outside 1..{n_classes - 1}")
        x, y = obj["tap_point_image"]
        if not (0 <= x < width and 0 <= y < height):
            report.add(sample_id, f"object {i} tap point {x, y} lies outside the image")


def _check_masks(report, sample_id, sess, meta, n_classes):
    labels, instances = sess.labels(sample_id), sess.instances(sample_id)
    if (labels is None) != (instances is None):
        report.add(sample_id, "labels.png and instances.png must both be present or both absent")
        return
    if labels is None:
        if sess.mask_status(sample_id) == "accepted":
            report.add(sample_id, "mask_status is accepted but there are no mask files")
        return
    expected_shape = (meta["image_size"][1], meta["image_size"][0])
    for name, array in (("labels.png", labels), ("instances.png", instances)):
        if array.dtype != np.uint8 or array.shape != expected_shape:
            report.add(sample_id, f"{name} must be 8-bit single-channel at the image size")
            return
    bad = set(np.unique(labels).tolist()) - set(range(n_classes)) - {fmt.IGNORE_LABEL}
    if bad:
        report.add(sample_id, f"labels.png holds unknown class ids {sorted(bad)}")
    if instances.max() > len(meta["objects"]):
        report.add(sample_id, "instances.png refers to more objects than meta.objects holds")


def _check_sample(report, sess, sample_id, n_classes):
    for filename in (fmt.META_FILE, fmt.IMAGE_FILE, fmt.DEPTH_FILE):
        if not sess.has(sample_id, filename):
            report.add(sample_id, f"missing {filename}")
    if sample_id in report.errors:
        return
    meta = sess.meta(sample_id)
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        report.add(sample_id, f"meta.json is missing {missing}")
        return
    if meta["sample_id"] != sample_id:
        report.add(sample_id, f"meta.sample_id is {meta['sample_id']!r}")

    image = sess.image(sample_id)
    if [image.shape[1], image.shape[0]] != list(meta["image_size"]):
        report.add(sample_id, f"image is {image.shape[1]}x{image.shape[0]}, meta says {meta['image_size']}")
    depth = sess.depth_mm(sample_id)
    if depth.dtype != np.uint16 or depth.ndim != 2:
        report.add(sample_id, "depth_mm.png must be a 16-bit single-channel PNG")
    elif [depth.shape[1], depth.shape[0]] != list(meta["depth_size"]):
        report.add(sample_id, f"depth is {depth.shape[1]}x{depth.shape[0]}, meta says {meta['depth_size']}")
    elif not (depth > 0).any():
        report.add(sample_id, "depth map has no readings at all")
    confidence = sess.confidence(sample_id)
    if confidence is not None and (confidence.shape != depth.shape or confidence.max() > 2):
        report.add(sample_id, "confidence.png must match the depth size and hold only 0/1/2")
    if not 0.0 <= meta["lens_position"] <= 1.0:
        report.add(sample_id, f"lens_position {meta['lens_position']} outside 0..1")

    _check_intrinsics(report, sample_id, meta)
    _check_objects(report, sample_id, meta, n_classes)
    _check_masks(report, sample_id, sess, meta, n_classes)

    report.n_objects += len(meta["objects"])
    report.n_background_only += bool(meta["background_only"])


def validate_session(path):
    """Validate one session zip (or extracted directory) and return a Report."""
    report = Report(path=str(path))
    try:
        sess = Session(path)
    except Exception as exc:  # unreadable zip or manifest
        report.add(None, f"cannot open session: {exc}")
        return report
    with sess:
        manifest = sess.manifest
        missing = [k for k in MANIFEST_KEYS if k not in manifest]
        if missing:
            report.add(None, f"manifest.json is missing {missing}")
            return report
        if manifest["format_version"] != fmt.FORMAT_VERSION:
            report.add(None, f"format_version {manifest['format_version']} is not {fmt.FORMAT_VERSION}")
        if manifest["split"] not in fmt.SPLITS:
            report.add(None, f"split {manifest['split']!r} is not one of {fmt.SPLITS}")
        classes = manifest["classes"]
        if len(classes) < 2 or classes[0] != "background":
            report.add(None, "classes must start with 'background' and name at least one class")
        sample_ids = manifest["sample_ids"]
        if len(set(sample_ids)) != len(sample_ids):
            report.add(None, "sample_ids holds duplicates")
        for sample_id, status in manifest["mask_status"].items():
            if status not in fmt.MASK_STATUSES:
                report.add(None, f"mask_status for {sample_id} is {status!r}")
        report.n_samples = len(sample_ids)
        for sample_id in sample_ids:
            try:
                _check_sample(report, sess, sample_id, len(classes))
            except Exception as exc:  # malformed JSON, corrupt image, wrong value types
                report.add(sample_id, f"could not be read: {exc!r}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="session zip(s) or extracted session directories")
    args = parser.parse_args(argv)
    reports = [validate_session(p) for p in args.paths]
    print("\n".join(r.format() for r in reports))
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
