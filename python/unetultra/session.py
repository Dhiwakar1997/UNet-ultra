"""Read a capture session (the data contract between the iOS app and the notebooks).

A session is a zip, or the same tree extracted to a directory:

    manifest.json
    samples/<sampleId>/{image.jpg, depth_mm.png, confidence.png, meta.json, labels.png, instances.png}
"""

import io
import json
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
