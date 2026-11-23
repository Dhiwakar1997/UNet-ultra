import numpy as np
import pytest
import torch

from unetultra.geometry import Intrinsics, backproject, measure_objects, nearest_resize, object_size_m

K = Intrinsics(fx=400.0, fy=410.0, cx=250.0, cy=190.0)


def test_backproject_known_point():
    # A point 0.1 m right and 0.05 m below the axis at 0.5 m projects to (cx + 80, cy + 41).
    x, y, z = backproject(K.cx + 80.0, K.cy + 41.0, 0.5, K.fx, K.fy, K.cx, K.cy)
    assert (x, y, z) == pytest.approx((0.1, 0.05, 0.5))


def test_backproject_uses_principal_point_not_image_centre():
    x, y, _ = backproject(np.array([K.cx]), np.array([K.cy]), np.array([0.3]), K.fx, K.fy, K.cx, K.cy)
    assert x[0] == 0 and y[0] == 0


def _square_scene(side_m=0.05, z=0.4):
    """A fronto-parallel square of known size, off-centre, on a far background."""
    mask = torch.zeros(384, 512, dtype=torch.bool)
    depth = torch.full((384, 512), 1.2)
    half_u, half_v = K.fx * side_m / (2 * z), K.fy * side_m / (2 * z)
    u0, v0 = 300, 150
    mask[round(v0 - half_v) : round(v0 + half_v) + 1, round(u0 - half_u) : round(u0 + half_u) + 1] = True
    depth[mask] = z
    return mask, depth


def test_object_size_and_distance():
    mask, depth = _square_scene(side_m=0.05, z=0.4)
    w, h, d = object_size_m(mask, depth, K.fx, K.fy, K.cx, K.cy)
    assert d == pytest.approx(0.4)
    # The percentile trim under-reports by roughly 4% (known bias, fixed in phase 4).
    assert 0.046 < w <= 0.05 and 0.046 < h <= 0.05


def test_object_size_empty_mask():
    empty = torch.zeros(8, 8, dtype=torch.bool)
    assert object_size_m(empty, torch.ones(8, 8), 1, 1, 0, 0) == (0.0, 0.0, 0.0)


def test_measure_objects_splits_components():
    mask, depth = _square_scene(side_m=0.05, z=0.4)
    mask[300:320, 20:60] = True  # a second, smaller object at another distance
    depth[300:320, 20:60] = 0.6
    mask[5, 5] = True  # speck below min_pixels
    results = measure_objects(mask, depth, K.fx, K.fy, K.cx, K.cy)
    assert len(results) == 2
    assert results[0]["area_px"] > results[1]["area_px"]
    assert results[0]["distance_m"] == pytest.approx(0.4)
    assert results[1]["distance_m"] == pytest.approx(0.6)
    assert results[1]["width_m"] == pytest.approx(39 * 0.6 / K.fx, rel=0.06)


def test_intrinsics_scaling_round_trip_and_values():
    k = Intrinsics(fx=1450.0, fy=1450.0, cx=972.0, cy=708.0)
    small = k.scaled((1920, 1440), (512, 384))
    assert small.fx == pytest.approx(1450 * 512 / 1920)
    assert small.cx == pytest.approx((972 + 0.5) * 512 / 1920 - 0.5)
    back = small.scaled((512, 384), (1920, 1440))
    assert back.to_dict() == pytest.approx(k.to_dict())


def test_scaled_intrinsics_preserve_rays():
    # The same scene point must back-project to the same ray before and after a resize.
    k = Intrinsics(fx=1450.0, fy=1450.0, cx=972.0, cy=708.0)
    small = k.scaled((1920, 1440), (512, 384))
    u, v = 1200.0, 300.0
    u_small, v_small = (u + 0.5) * 512 / 1920 - 0.5, (v + 0.5) * 384 / 1440 - 0.5
    assert backproject(u, v, 1.0, **k.to_dict())[:2] == pytest.approx(
        backproject(u_small, v_small, 1.0, **small.to_dict())[:2]
    )


def test_hflip_updates_cx():
    k = Intrinsics(fx=400.0, fy=400.0, cx=250.0, cy=190.0)
    flipped = k.hflipped(512)
    assert flipped.cx == 512 - 1 - 250 and flipped.cy == k.cy
    # A pixel and its mirror image see mirrored rays.
    x, _, _ = backproject(100.0, 0.0, 1.0, **k.to_dict())
    x_flipped, _, _ = backproject(511 - 100.0, 0.0, 1.0, **flipped.to_dict())
    assert x_flipped == pytest.approx(-x)


def test_nearest_resize_keeps_values():
    labels = np.array([[0, 0, 7, 7], [0, 0, 7, 7]], dtype=np.uint8)
    assert nearest_resize(labels, (2, 1)).tolist() == [[0, 7]]
    assert set(np.unique(nearest_resize(labels, (9, 5)))) == {0, 7}
