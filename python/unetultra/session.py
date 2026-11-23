"""Read a capture session (the data contract between the iOS app and the notebooks).

A session is a zip, or the same tree extracted to a directory:

    manifest.json
    samples/<sampleId>/{image.jpg, depth_mm.png, confidence.png, meta.json, labels.png, instances.png}
"""

import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

FORMAT_VERSION = 1
SPLITS = ("train", "test")
MASK_STATUSES = ("pending", "accepted", "rejected")
IGNORE_LABEL = 255
HIGH_CONFIDENCE = 2

IMAGE_FILE = "image.jpg"
DEPTH_FILE = "depth_mm.png"
CONFIDENCE_FILE = "confidence.png"
META_FILE = "meta.json"
LABELS_FILE = "labels.png"
INSTANCES_FILE = "instances.png"


class Session:
    """Read-only view of one capture session stored as a zip or a directory."""

    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_dir():
            self._zip = None
            self._names = {p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file()}
        else:
            self._zip = zipfile.ZipFile(self.path)
            self._names = {n for n in self._zip.namelist() if not n.endswith("/")}
        self.manifest = json.loads(self.read_bytes("manifest.json"))

    def close(self):
        if self._zip is not None:
            self._zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- raw access -------------------------------------------------------------------------

    def exists(self, name):
        return name in self._names

    def read_bytes(self, name):
        if name not in self._names:
            raise FileNotFoundError(f"{name} not in session {self.path}")
        if self._zip is not None:
            return self._zip.read(name)
        return (self.path / name).read_bytes()

    def _array(self, name):
        return np.array(Image.open(io.BytesIO(self.read_bytes(name))))

    @staticmethod
    def sample_path(sample_id, filename):
        return f"samples/{sample_id}/{filename}"

    def has(self, sample_id, filename):
        return self.exists(self.sample_path(sample_id, filename))

    # -- manifest ---------------------------------------------------------------------------

    @property
    def session_id(self):
        return self.manifest["session_id"]

    @property
    def split(self):
        return self.manifest["split"]

    @property
    def classes(self):
        return self.manifest["classes"]

    @property
    def sample_ids(self):
        return self.manifest["sample_ids"]

    def mask_status(self, sample_id):
        return self.manifest.get("mask_status", {}).get(sample_id, "pending")

    # -- per-sample -------------------------------------------------------------------------

    def meta(self, sample_id):
        return json.loads(self.read_bytes(self.sample_path(sample_id, META_FILE)))

    def image(self, sample_id):
        """RGB image as uint8 [H, W, 3], in sensor orientation."""
        data = self.read_bytes(self.sample_path(sample_id, IMAGE_FILE))
        return np.array(Image.open(io.BytesIO(data)).convert("RGB"))

    def depth_mm(self, sample_id):
        """Raw depth as uint16 [h, w] in millimetres; 0 means no reading."""
        return self._array(self.sample_path(sample_id, DEPTH_FILE))

    def depth_m(self, sample_id):
        """Depth as float32 [h, w] in metres; 0 means no reading."""
        return self.depth_mm(sample_id).astype(np.float32) / 1000.0

    def confidence(self, sample_id):
        """LiDAR confidence as uint8 [h, w] (0/1/2), or None if the session has none."""
        name = self.sample_path(sample_id, CONFIDENCE_FILE)
        return self._array(name) if self.exists(name) else None

    def depth_valid(self, sample_id):
        """Boolean [h, w]: depth > 0 and, where a confidence map exists, high confidence."""
        valid = self.depth_mm(sample_id) > 0
        confidence = self.confidence(sample_id)
        if confidence is not None:
            valid &= confidence >= HIGH_CONFIDENCE
        return valid

    def labels(self, sample_id):
        """Class id per pixel (0 = background, 255 = ignore), or None before mask review."""
        name = self.sample_path(sample_id, LABELS_FILE)
        return self._array(name) if self.exists(name) else None

    def instances(self, sample_id):
        """Object index per pixel (0 = none, 1..N in meta.objects order), or None."""
        name = self.sample_path(sample_id, INSTANCES_FILE)
        return self._array(name) if self.exists(name) else None


def image_to_depth_point(point, image_size, depth_size):
    """Map an image pixel (x, y) to depth-map pixel coordinates (pixel-centre convention)."""
    return (
        (point[0] + 0.5) * depth_size[0] / image_size[0] - 0.5,
        (point[1] + 0.5) * depth_size[1] / image_size[1] - 0.5,
    )


def lidar_at_point(depth_m, valid, point, image_size, window=2):
    """Median LiDAR depth (m) over valid pixels in a (2*window+1)^2 box around an image point.

    Returns (distance_m, n_valid); distance is nan when no valid pixel falls in the box."""
    h, w = depth_m.shape
    u, v = image_to_depth_point(point, image_size, (w, h))
    u, v = int(round(u)), int(round(v))
    rows = slice(max(v - window, 0), min(v + window + 1, h))
    cols = slice(max(u - window, 0), min(u + window + 1, w))
    values = depth_m[rows, cols][valid[rows, cols]]
    if values.size == 0:
        return float("nan"), 0
    return float(np.median(values)), int(values.size)


def png_bytes(array):
    """Encode a uint8 or uint16 [H, W] array as PNG."""
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def update_session(path, files, manifest=None):
    """Add or replace files in a session (zip or directory), optionally replacing the manifest.

    `files` maps names inside the session to bytes. A zip is rewritten to a temporary file and
    swapped in at the end, so an interrupted write never leaves a half-written session."""
    files = dict(files)
    if manifest is not None:
        files["manifest.json"] = json.dumps(manifest, indent=2).encode()
    path = Path(path)
    if path.is_dir():
        for name, data in files.items():
            target = path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return
    fd, tmp = tempfile.mkstemp(suffix=".zip", dir=path.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zout:
            for info in zin.infolist():
                if info.filename not in files and not info.is_dir():
                    zout.writestr(info.filename, zin.read(info.filename))
            for name, data in files.items():
                zout.writestr(name, data)
        shutil.move(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
