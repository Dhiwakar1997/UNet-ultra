"""Pinhole geometry: intrinsics handling, back-projection and object measurement.

Convention: pixel (u, v) refers to the centre of the pixel in column u, row v, so integer
coordinates are pixel centres. Size and distance are computed here, never learned.
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in pixels, valid for one specific image size."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_dict(cls, d):
        return cls(float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]))

    def to_dict(self):
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}

    def scaled(self, from_size, to_size):
        """Intrinsics after resizing an image from from_size to to_size, both (W, H).

        The principal point uses the pixel-centre mapping u' = (u + 0.5) * s - 0.5, which is
        what an area-aligned resize does to pixel centres.
        """
        sx = to_size[0] / from_size[0]
        sy = to_size[1] / from_size[1]
        return Intrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=(self.cx + 0.5) * sx - 0.5,
            cy=(self.cy + 0.5) * sy - 0.5,
        )

    def hflipped(self, width):
        """Intrinsics after a horizontal flip of an image `width` pixels wide."""
        return Intrinsics(self.fx, self.fy, width - 1 - self.cx, self.cy)


def backproject(u, v, z, fx, fy, cx, cy):
    """Pixel coordinates and depth along the optical axis -> camera-space (X, Y, Z).

    Works on numpy arrays or torch tensors. Units of X and Y follow the units of z.
    """
    return (u - cx) * z / fx, (v - cy) * z / fy, z


def object_size_m(mask, depth, fx, fy, cx, cy):
    """mask, depth: [H, W] at the model's resolution; intrinsics scaled to the same resolution.
    Returns (width_m, height_m, distance_m) for the pixels in mask.

    Known bias: the 2nd-98th percentile trim, plus measuring between pixel centres,
    under-reports size by about 4%. Revisited in phase 4.
    """
    v, u = torch.nonzero(mask, as_tuple=True)
    if len(u) == 0:
        return 0.0, 0.0, 0.0
    Z = depth[v, u]
    X, Y, _ = backproject(u.float(), v.float(), Z, fx, fy, cx, cy)
    width = torch.quantile(X, 0.98) - torch.quantile(X, 0.02)
    height = torch.quantile(Y, 0.98) - torch.quantile(Y, 0.02)
    return width.item(), height.item(), Z.median().item()


def measure_objects(mask, depth, fx, fy, cx, cy, min_pixels=30):
    """Split a binary mask into connected components and measure each one.
    Returns a list of dicts sorted by area, largest first."""
    components, n = ndimage.label(mask.cpu().numpy())
    results = []
    for i in range(1, n + 1):
        blob = torch.from_numpy(components == i)
        area = int(blob.sum())
        if area < min_pixels:
            continue
        w, h, d = object_size_m(blob, depth, fx, fy, cx, cy)
        results.append({"id": i, "area_px": area, "width_m": w, "height_m": h, "distance_m": d})
    return sorted(results, key=lambda r: r["area_px"], reverse=True)


def nearest_resize(array, size):
    """Nearest-neighbour resize of a [H, W] array to size (W, H), sampling at pixel centres.

    Used for depth, labels and instances, where interpolation would invent values.
    """
    h, w = array.shape[:2]
    cols = np.minimum(((np.arange(size[0]) + 0.5) * w / size[0]).astype(int), w - 1)
    rows = np.minimum(((np.arange(size[1]) + 0.5) * h / size[1]).astype(int), h - 1)
    return array[rows[:, None], cols[None, :]]
