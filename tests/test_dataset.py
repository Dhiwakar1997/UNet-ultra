import numpy as np
import pytest
import torch

from unetultra.calibration import fit_from_sessions
from unetultra.dataset import SessionDataset, augment, meta_vector
from unetultra.fake_session import make_fake_session
from unetultra.session import Session
from unetultra.train import split_sessions


@pytest.fixture(scope="module")
def calibration(fake_zips):
    return fit_from_sessions(fake_zips["train"])[0]


def test_items_have_model_shapes_and_scaled_intrinsics(fake_zips, calibration):
    dataset = SessionDataset(fake_zips["train"][:1], calibration)
    assert len(dataset) == 9 and dataset.n_classes == 1
    item = dataset[0]
    assert item["image"].shape == (3, 512, 512) and 0 <= item["image"].min() and item["image"].max() <= 1
    for key in ("labels", "instances", "depth", "depth_valid"):
        assert item[key].shape == (512, 512)
    with Session(fake_zips["train"][0]) as sess:
        meta = sess.meta(item["sample_id"])
    # 512x384 frames: the centre square is 384 wide, starting at x0 = 64.
    k, s = meta["intrinsics_image"], 512 / 384
    fx = k["fx"] * s
    assert item["intrinsics"].tolist() == pytest.approx(
        [fx, k["fy"] * s, (k["cx"] - 64 + 0.5) * s - 0.5, (k["cy"] + 0.5) * s - 0.5])
    focus = calibration.focus_distance_m(meta["lens_position"])
    assert item["meta"].numpy() == pytest.approx(meta_vector(focus, fx))
    assert item["meta"][0] == pytest.approx(np.log(item["focus_distance_m"]))


def test_multi_object_and_background_only_samples(fake_zips, calibration):
    dataset = SessionDataset(fake_zips["train"], calibration)
    counts = [int(dataset[i]["instances"].max()) for i in range(len(dataset))]
    assert 0 in counts and 2 in counts
    empty = dataset[counts.index(0)]
    assert not empty["labels"].any()
    double = dataset[counts.index(2)]
    assert set(double["labels"].unique().tolist()) == {0, 1}
    # Depth on each object matches the LiDAR distance recorded at its tap.
    meta = dataset.sample_meta(counts.index(2))
    for index, obj in enumerate(meta["objects"], start=1):
        region = (double["instances"] == index) & double["depth_valid"]
        assert float(double["depth"][region].median()) == pytest.approx(obj["lidar_distance_at_tap_m"], abs=0.003)


def test_depth_and_image_crops_line_up(fake_zips, calibration):
    """After cropping, the depth discontinuity around each object sits on the mask boundary."""
    dataset = SessionDataset(fake_zips["train"], calibration)
    checked = 0
    for index in range(len(dataset)):
        item = dataset[index]
        if item["instances"].max() == 0:
            continue
        near = item["depth"] < 0.75  # objects are within 0.7 m, the wall is beyond 0.8 m
        mask = item["instances"] > 0
        disagreement = (near ^ mask) & item["depth_valid"]
        # Only a thin band at the edges may disagree (depth is a 2-3x coarser grid than the mask).
        assert disagreement.sum() < 0.35 * mask.sum()
        # The tap, transformed with the crop, lands on its object.
        from unetultra.geometry import CenterSquareCrop
        meta = dataset.sample_meta(index)
        crop = CenterSquareCrop.for_size(meta["image_size"])
        for number, obj in enumerate(meta["objects"], start=1):
            u, v = crop.point(*obj["tap_point_image"])
            assert int(item["instances"][round(v), round(u)]) == number
            checked += 1
    assert checked >= 15


def test_samples_tapped_outside_the_square_are_reported_not_dropped(fake_zips, calibration):
    dataset = SessionDataset(fake_zips["train"], calibration)
    assert len(dataset) == 27 and len(dataset.excluded) == 3 and dataset.skipped == 0
    assert all(reason == "excluded_outside_crop" and sample_id == "0003" for _, sample_id, reason in dataset.excluded)


def test_mismatched_aspect_ratio_is_an_error():
    from unetultra.geometry import check_same_aspect

    check_same_aspect((1920, 1440), (256, 192))
    with pytest.raises(ValueError, match="aspect"):
        check_same_aspect((1920, 1080), (256, 192))


def test_only_accepted_samples_are_used(tmp_path, calibration):
    pending = make_fake_session(tmp_path / "pending.zip", n_samples=4, image_size=(256, 192))
    dataset = SessionDataset([pending], calibration)
    assert len(dataset) == 0 and dataset.skipped == 4


def test_image_only_ablation_zeroes_meta(fake_zips, calibration):
    item = SessionDataset(fake_zips["train"][:1], calibration, use_meta=False)[1]
    assert not item["meta"].any()


def test_flip_updates_cx_and_keeps_geometry(fake_zips, calibration):
    dataset = SessionDataset(fake_zips["train"][:1], calibration)
    original = dataset[0]
    generator = torch.Generator().manual_seed(0)
    flipped = None
    for _ in range(20):
        candidate = augment(dataset[0], generator)
        if candidate["intrinsics"][2] != original["intrinsics"][2]:
            flipped = candidate
            break
    assert flipped is not None
    assert float(flipped["intrinsics"][2]) == pytest.approx(511 - float(original["intrinsics"][2]))
    assert torch.equal(flipped["labels"], original["labels"].flip(-1))
    assert torch.equal(flipped["depth"], original["depth"].flip(-1))
    assert not torch.equal(flipped["image"], original["image"].flip(-1))  # photometric jitter applied


def test_split_is_by_session_and_refuses_test_data(fake_zips):
    train, val = split_sessions(fake_zips["train"])
    assert len(val) == 1 and len(train) == 2 and not set(train) & set(val)
    with pytest.raises(ValueError, match="at least two"):
        split_sessions(fake_zips["train"][:1])
    with pytest.raises(ValueError, match="must not be used for training"):
        split_sessions(fake_zips["train"] + [fake_zips["test"]])
