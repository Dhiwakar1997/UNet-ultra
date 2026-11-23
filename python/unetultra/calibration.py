"""Per-device lens-position -> focus-distance calibration.

`lensPosition` (0..1) is not a distance and differs between devices. We fit a monotonic curve
on the training captures, against LiDAR depth at the focused tap point, and the model only ever
sees the calibrated distance.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .session import Session, lidar_at_point

FORMAT_VERSION = 1
MIN_TAP_PIXELS = 5


@dataclass
class LensCalibration:
    """Monotonic knots (lens_position ascending) mapping the lens reading to distance in metres.

    Between knots the curve is linear in 1/distance, which is roughly how lens travel behaves;
    outside the fitted range it is clamped to the end values."""

    lens_position: np.ndarray
    distance_m: np.ndarray
    device_model: str = ""
    n_samples: int = 0
    rmse_m: float = float("nan")
    created_at: str = ""

    def focus_distance_m(self, lens_position):
        """Calibrated focus distance (m) for a lens reading or an array of readings."""
        inverse = np.interp(lens_position, self.lens_position, 1.0 / self.distance_m)
        return 1.0 / inverse

    def to_dict(self):
        return {
            "format_version": FORMAT_VERSION,
            "kind": "lens_position_to_distance",
            "interpolation": "linear_in_inverse_distance",
            "device_model": self.device_model,
            "lens_position": [float(x) for x in self.lens_position],
            "distance_m": [float(x) for x in self.distance_m],
            "n_samples": int(self.n_samples),
            "rmse_m": float(self.rmse_m),
            "created_at": self.created_at,
        }

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        return Path(path)

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(
            np.asarray(d["lens_position"], dtype=float), np.asarray(d["distance_m"], dtype=float),
            d.get("device_model", ""), d.get("n_samples", 0), d.get("rmse_m", float("nan")), d.get("created_at", ""),
        )


def collect_calibration_points(paths):
    """Gather (lens_position, LiDAR distance at the focused tap) pairs from session zips.

    Only samples whose focus settled on a tapped object are used; distance is the median of valid,
    high-confidence depth in a small window at the tap. Returns a dict of arrays plus device models."""
    lens, distance, sample, devices = [], [], [], set()
    for path in paths:
        with Session(path) as sess:
            devices.add(sess.manifest.get("device_model", ""))
            for sample_id in sess.sample_ids:
                meta = sess.meta(sample_id)
                if not meta["objects"] or not meta.get("focus_settled", True):
                    continue
                obj = meta["objects"][meta.get("focus_tap_index", 0)]
                d, n = lidar_at_point(sess.depth_m(sample_id), sess.depth_valid(sample_id),
                                      obj["tap_point_image"], meta["image_size"])
                if n >= MIN_TAP_PIXELS and np.isfinite(d):
                    lens.append(meta["lens_position"])
                    distance.append(d)
                    sample.append(f"{sess.session_id}/{sample_id}")
    return {"lens_position": np.asarray(lens), "distance_m": np.asarray(distance),
            "sample": sample, "device_models": sorted(devices)}


def fit_calibration(lens_position, distance_m, device_model=""):
    """Fit a monotonic lens-position -> distance curve with isotonic regression on 1/distance."""
    from sklearn.isotonic import IsotonicRegression

    lens_position = np.asarray(lens_position, dtype=float)
    distance_m = np.asarray(distance_m, dtype=float)
    if len(lens_position) < 3:
        raise ValueError("need at least 3 focused samples to fit a lens calibration")
    iso = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(lens_position, 1.0 / distance_m)
    knots_x = np.asarray(iso.X_thresholds_, dtype=float)
    knots_inverse = np.asarray(iso.y_thresholds_, dtype=float)
    calibration = LensCalibration(
        knots_x, 1.0 / knots_inverse, device_model, len(lens_position),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    residual = calibration.focus_distance_m(lens_position) - distance_m
    calibration.rmse_m = float(np.sqrt(np.mean(residual**2)))
    return calibration


def fit_from_sessions(paths):
    """Collect points from training sessions and fit. Returns (calibration, points)."""
    points = collect_calibration_points(paths)
    device = ", ".join(points["device_models"])
    return fit_calibration(points["lens_position"], points["distance_m"], device), points
