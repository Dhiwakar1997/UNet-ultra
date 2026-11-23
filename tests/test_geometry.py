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


def test_robust_size_removes_the_known_bias():
    from unetultra.geometry import object_size_robust_m

    mask, depth = _square_scene(side_m=0.05, z=0.4)
    w, h, d = object_size_robust_m(mask, depth, K.fx, K.fy, K.cx, K.cy)
    # The exact answer is the mask's pixel count along each axis times the size of one pixel.
    columns, rows = int(mask.any(dim=0).sum()), int(mask.any(dim=1).sum())
    assert w == pytest.approx(columns * 0.4 / K.fx, rel=1e-3) and h == pytest.approx(rows * 0.4 / K.fy, rel=1e-3)
    assert (w, h) == pytest.approx((0.05, 0.05), rel=0.04) and d == pytest.approx(0.4)
    raw_w, _, _ = object_size_m(mask, depth, K.fx, K.fy, K.cx, K.cy)
    assert abs(w - 0.05) < abs(raw_w - 0.05)


def test_robust_size_survives_background_depth_on_the_edges():
    from unetultra.geometry import object_size_robust_m

    mask, depth = _square_scene(side_m=0.05, z=0.4)
    rows, cols = torch.nonzero(mask, as_tuple=True)
    depth[rows.min() : rows.max() + 1, cols.max() - 2 : cols.max() + 1] = 1.2  # edge picked up the wall
    raw_w, _, _ = object_size_m(mask, depth, K.fx, K.fy, K.cx, K.cy)
    w, _, d = object_size_robust_m(mask, depth, K.fx, K.fy, K.cx, K.cy)
    assert raw_w > 0.08  # the reference method is thrown far off
    assert w == pytest.approx(0.05, rel=0.04) and d == pytest.approx(0.4)


# ---- centre-square crop (the model's view of a frame) ----

def test_crop_maps_centre_and_edges():
    from unetultra.geometry import CenterSquareCrop

    crop = CenterSquareCrop.for_size((1920, 1440))
    assert (crop.x0, crop.y0, crop.side, crop.out) == (240, 0, 1440, 512)
    assert crop.point(959.5, 719.5) == pytest.approx((255.5, 255.5))  # image centre
    assert crop.point(239.5, -0.5) == pytest.approx((-0.5, -0.5))  # left/top edge of the crop
    assert crop.point(1679.5, 1439.5) == pytest.approx((511.5, 511.5))  # right/bottom edge
    assert crop.contains(240, 0) and crop.contains(1679, 1439)
    assert not crop.contains(239, 700) and not crop.contains(1680, 700)


def test_crop_intrinsics_follow_the_formula():
    from unetultra.geometry import CenterSquareCrop

    k = CenterSquareCrop.for_size((1920, 1440)).intrinsics(Intrinsics(1450.0, 1452.0, 972.0, 708.0))
    s = 512 / 1440
    assert k.to_dict() == pytest.approx(
        {"fx": 1450 * s, "fy": 1452 * s, "cx": (972 - 240 + 0.5) * s - 0.5, "cy": (708 + 0.5) * s - 0.5})


def test_crop_of_a_portrait_or_depth_sized_map():
    from unetultra.geometry import CenterSquareCrop

    depth_crop = CenterSquareCrop.for_size((256, 192))
    assert (depth_crop.x0, depth_crop.y0, depth_crop.side) == (32, 0, 192)
    grid = np.arange(192 * 256).reshape(192, 256)
    assert depth_crop.crop(grid).shape == (192, 192) and depth_crop.crop(grid)[0, 0] == 32
    assert depth_crop.resize_nearest(grid).shape == (512, 512)
    tall = CenterSquareCrop.for_size((1440, 1920))
    assert (tall.x0, tall.y0, tall.side) == (0, 240, 1440)


def test_object_measures_the_same_before_and_after_the_crop():
    """A square of known size and distance: same answer on the full frame and on the model's crop."""
    from unetultra.geometry import CenterSquareCrop, object_size_robust_m

    k = Intrinsics(fx=1450.0, fy=1450.0, cx=972.0, cy=708.0)
    side_m, z = 0.055, 0.35
    full_mask = np.zeros((1440, 1920), dtype=bool)
    half = k.fx * side_m / (2 * z)
    u0, v0 = 1100, 600
    full_mask[round(v0 - half) : round(v0 + half) + 1, round(u0 - half) : round(u0 + half) + 1] = True
    full_depth = np.where(full_mask, z, 1.3).astype(np.float32)
    before = object_size_robust_m(torch.from_numpy(full_mask), torch.from_numpy(full_depth), k.fx, k.fy, k.cx, k.cy)

    crop = CenterSquareCrop.for_size((1920, 1440))
    k_crop = crop.intrinsics(k)
    mask = torch.from_numpy(crop.resize_nearest(full_mask))
    depth = torch.from_numpy(crop.resize_nearest(full_depth))
    after = object_size_robust_m(mask, depth, k_crop.fx, k_crop.fy, k_crop.cx, k_crop.cy)
    one_model_pixel = z / k_crop.fx
    assert after[2] == pytest.approx(before[2]) == pytest.approx(z)
    # Resampling can move each edge by up to half a model pixel, so allow 1.5 pixels in total.
    assert after[0] == pytest.approx(before[0], abs=1.5 * one_model_pixel)
    assert after[1] == pytest.approx(before[1], abs=1.5 * one_model_pixel)
    assert after[0] == pytest.approx(side_m, abs=one_model_pixel)
    # The object's centre back-projects to the same 3D point through either set of intrinsics.
    rows, cols = np.nonzero(full_mask)
    v_c, u_c = torch.nonzero(mask, as_tuple=True)
    x_full, y_full, _ = backproject(cols.mean(), rows.mean(), z, **k.to_dict())
    x_crop, y_crop, _ = backproject(float(u_c.float().mean()), float(v_c.float().mean()), z, **k_crop.to_dict())
    assert (x_crop, y_crop) == pytest.approx((x_full, y_full), abs=one_model_pixel)
