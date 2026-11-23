"""Training dataset: accepted samples from session zips, resized to the model input.

Images are resized to 512x384 with the intrinsics scaled to match; depth, labels and instances
are resized with nearest neighbour. The model's inputs are the image and the meta vector only.
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from .geometry import Intrinsics, nearest_resize
from .model import INPUT_HEIGHT, INPUT_WIDTH
from .session import IGNORE_LABEL, Session

MODEL_SIZE = (INPUT_WIDTH, INPUT_HEIGHT)


def meta_vector(focus_distance_m, fx_model):
    """The model's meta input: [log(calibrated focus distance in m), fx at model size / 1000 - 1]."""
    return np.array([np.log(focus_distance_m), fx_model / 1000.0 - 1.0], dtype=np.float32)


def resize_image(image, size=MODEL_SIZE):
    """Resize a uint8 RGB image to the model input size (antialiased bilinear)."""
    return np.asarray(Image.fromarray(image).resize(size, Image.BILINEAR))


def image_to_tensor(image):
    """uint8 [H, W, 3] -> float [3, H, W] in 0..1. This is the only normalisation the model uses."""
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0


def augment(item, generator):
    """Photometric jitter, noise, blur and horizontal flip. Geometry stays exact: the flip
    updates cx, and nothing else moves pixels."""
    def rand(lo, hi):
        return float(torch.empty(1).uniform_(lo, hi, generator=generator))

    image = item["image"]
    image = image * rand(0.75, 1.25)
    mean = image.mean()
    image = (image - mean) * rand(0.75, 1.25) + mean
    grey = image.mean(dim=0, keepdim=True)
    image = (image - grey) * rand(0.8, 1.2) + grey
    image = image * torch.empty(3, 1, 1).uniform_(0.95, 1.05, generator=generator)
    image = image + torch.randn(image.shape, generator=generator) * rand(0.0, 0.03)
    if rand(0, 1) < 0.2:
        image = TF.gaussian_blur(image, kernel_size=5, sigma=rand(0.3, 1.5))
    item["image"] = image.clamp(0, 1)
    if rand(0, 1) < 0.5:
        for key in ("image", "labels", "instances", "depth", "depth_valid"):
            item[key] = item[key].flip(-1)
        width = item["image"].shape[-1]
        item["intrinsics"] = item["intrinsics"].clone()
        item["intrinsics"][2] = width - 1 - item["intrinsics"][2]
    return item


class SessionDataset(Dataset):
    """Accepted samples from one or more session zips.

    Each item is a dict: image [3, H, W], meta [2], labels [H, W] (class id, 255 = ignore),
    instances [H, W], depth [H, W] in metres (training label only), depth_valid [H, W],
    intrinsics [fx, fy, cx, cy] at the model size, plus focus_distance_m, lens_position and ids.
    With use_meta=False the meta vector is zeroed (the image-only ablation).
    """

    def __init__(self, paths, calibration, augment=False, use_meta=True, size=MODEL_SIZE, seed=0):
        self.paths = [str(p) for p in paths]
        self.calibration = calibration
        self.augment = augment
        self.use_meta = use_meta
        self.size = tuple(size)
        self.generator = torch.Generator().manual_seed(seed)
        self.refs, self.classes, self.skipped = [], None, 0
        for path in self.paths:
            with Session(path) as sess:
                if self.classes is None:
                    self.classes = list(sess.classes)
                elif list(sess.classes) != self.classes:
                    raise ValueError(f"{path} has classes {sess.classes}, expected {self.classes}")
                for sample_id in sess.sample_ids:
                    if sess.mask_status(sample_id) == "accepted" and sess.has(sample_id, "labels.png"):
                        self.refs.append((path, sample_id))
                    else:
                        self.skipped += 1
        self._cache = {}

    @property
    def n_classes(self):
        """Number of object classes, not counting background."""
        return len(self.classes) - 1

    def __len__(self):
        return len(self.refs)

    def _load(self, index):
        path, sample_id = self.refs[index]
        with Session(path) as sess:
            meta = sess.meta(sample_id)
            image_size = tuple(meta["image_size"])
            k = Intrinsics.from_dict(meta["intrinsics_image"]).scaled(image_size, self.size)
            focus = float(self.calibration.focus_distance_m(meta["lens_position"]))
            return {
                "image": resize_image(sess.image(sample_id), self.size),
                "labels": nearest_resize(sess.labels(sample_id), self.size),
                "instances": nearest_resize(sess.instances(sample_id), self.size),
                "depth": nearest_resize(sess.depth_m(sample_id), self.size),
                "depth_valid": nearest_resize(sess.depth_valid(sample_id), self.size),
                "intrinsics": np.array([k.fx, k.fy, k.cx, k.cy], dtype=np.float32),
                "meta": meta_vector(focus, k.fx),
                "focus_distance_m": focus,
                "lens_position": float(meta["lens_position"]),
                "focus_tap_index": int(meta.get("focus_tap_index", 0)),
                "session_id": sess.session_id,
                "sample_id": sample_id,
            }

    def sample_meta(self, index):
        """The sample's full meta.json (for measured sizes and notes during evaluation)."""
        path, sample_id = self.refs[index]
        with Session(path) as sess:
            return sess.meta(sample_id)

    def __getitem__(self, index):
        if index not in self._cache:
            self._cache[index] = self._load(index)
        raw = self._cache[index]
        item = {
            "image": image_to_tensor(raw["image"]),
            "labels": torch.from_numpy(raw["labels"].astype(np.int64)),
            "instances": torch.from_numpy(raw["instances"].astype(np.int64)),
            "depth": torch.from_numpy(raw["depth"].astype(np.float32)),
            "depth_valid": torch.from_numpy(raw["depth_valid"].copy()),
            "intrinsics": torch.from_numpy(raw["intrinsics"].copy()),
            "meta": torch.from_numpy(raw["meta"].copy()),
            "focus_distance_m": raw["focus_distance_m"],
            "lens_position": raw["lens_position"],
            "focus_tap_index": raw["focus_tap_index"],
            "session_id": raw["session_id"],
            "sample_id": raw["sample_id"],
            "index": index,
        }
        if not self.use_meta:
            item["meta"] = torch.zeros_like(item["meta"])
        if self.augment:
            item = augment(item, self.generator)
        return item


__all__ = ["SessionDataset", "meta_vector", "resize_image", "image_to_tensor", "augment", "MODEL_SIZE", "IGNORE_LABEL"]
