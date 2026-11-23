import json
import zipfile

import numpy as np
import pytest
import torch

from unetultra.fake_session import fake_lens_position, make_fake_session
from unetultra.geometry import Intrinsics, measure_objects, nearest_resize
from unetultra.session import Session
from unetultra.validate import main, validate_session


def test_fake_session_is_valid(fake_zip):
    report = validate_session(fake_zip)
    assert report.ok, report.format()
    assert report.n_samples == 8 and report.n_background_only >= 1
    assert report.excluded_outside_crop == ["0003"] and "excluded_outside_crop" in report.format()
    assert main([str(fake_zip)]) == 0


def test_session_contents(fake_zip):
    with Session(fake_zip) as sess:
        assert sess.classes == ["background", "airpods_case"] and sess.split == "train"
        counts = [len(sess.meta(s)["objects"]) for s in sess.sample_ids]
        assert 0 in counts and 2 in counts
        sample_id = sess.sample_ids[counts.index(2)]
        meta = sess.meta(sample_id)
        assert sess.image(sample_id).shape == (384, 512, 3)
        assert sess.depth_mm(sample_id).dtype == np.uint16
        assert sess.depth_valid(sample_id).mean() > 0.95
        assert set(np.unique(sess.instances(sample_id))) == {0, 1, 2}
        background = sess.sample_ids[counts.index(0)]
        assert sess.meta(background)["background_only"] and not sess.labels(background).any()
        assert meta["lens_position"] == pytest.approx(
            fake_lens_position(meta["objects"][0]["lidar_distance_at_tap_m"]), abs=0.02
        )


def test_depth_and_masks_agree_with_stated_truth(fake_zip):
    """Depth at the tap, and size from mask + depth + intrinsics, match the known shapes."""
    with Session(fake_zip) as sess:
        checked = 0
        for sample_id in sess.sample_ids:
            meta = sess.meta(sample_id)
            k = Intrinsics.from_dict(meta["intrinsics_image"])
            k_depth = Intrinsics.from_dict(meta["intrinsics_depth"])
            depth = sess.depth_m(sample_id)
            depth_size = tuple(meta["depth_size"])
            for index, obj in enumerate(meta["objects"], start=1):
                u, v = obj["tap_point_image"]
                u_d = round((u + 0.5) * k_depth.fx / k.fx - 0.5)
                v_d = round((v + 0.5) * k_depth.fy / k.fy - 0.5)
                window = depth[v_d - 1 : v_d + 2, u_d - 1 : u_d + 2]
                distance_m = obj["true_distance_cm"] / 100
                assert np.median(window) == pytest.approx(distance_m, abs=0.003)

                # Measure at the depth map's resolution, where mask and depth sample the same points.
                mask = torch.from_numpy(nearest_resize(sess.instances(sample_id), depth_size) == index)
                (result,) = measure_objects(
                    mask, torch.from_numpy(depth), k_depth.fx, k_depth.fy, k_depth.cx, k_depth.cy
                )
                assert result["distance_m"] == pytest.approx(distance_m, abs=0.003)
                # Allow the known ~4% trim bias (more for ellipses) plus two depth pixels.
                for axis, f in (("width", k_depth.fx), ("height", k_depth.fy)):
                    truth_cm = obj["true_size_cm"][axis]
                    tolerance_cm = 0.08 * truth_cm + 2 * distance_m / f * 100
                    assert result[f"{axis}_m"] * 100 == pytest.approx(truth_cm, abs=tolerance_cm)
                checked += 1
        assert checked >= 5


def test_generation_is_deterministic_and_masks_optional(tmp_path):
    a = make_fake_session(tmp_path / "a.zip", n_samples=4, image_size=(256, 192), seed=3)
    b = make_fake_session(tmp_path / "b.zip", n_samples=4, image_size=(256, 192), seed=3, with_masks=True)
    with Session(a) as sa, Session(b) as sb:
        assert sa.labels("0000") is None and sa.mask_status("0000") == "pending"
        assert sb.mask_status("0000") == "accepted"
        assert sa.meta("0001") == sb.meta("0001")
        assert (sa.depth_mm("0001") == sb.depth_mm("0001")).all()
    assert validate_session(a).ok


def test_multi_class_session(tmp_path):
    path = make_fake_session(
        tmp_path / "m.zip", n_samples=10, image_size=(256, 192), with_masks=True,
        classes=("background", "airpods_case", "mug", "keys"),
    )
    assert validate_session(path).ok
    with Session(path) as sess:
        ids = {o["class_id"] for s in sess.sample_ids for o in sess.meta(s)["objects"]}
    assert len(ids) > 1 and ids <= {1, 2, 3}


def _rewrite(src, dst, edit_meta=None, drop=None):
    """Copy a session zip, editing sample 0001's meta or dropping a file."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for name in zin.namelist():
            data = zin.read(name)
            if name == drop:
                continue
            if edit_meta and name == "samples/0001/meta.json":
                meta = json.loads(data)
                edit_meta(meta)
                data = json.dumps(meta)
            zout.writestr(name, data)
    return dst


def test_validator_catches_broken_samples(fake_zip, tmp_path):
    def swap_axes(meta):  # the classic bug: image rotated but intrinsics not
        meta["image_size"] = meta["image_size"][::-1]

    def tap_outside(meta):
        meta["objects"][0]["tap_point_image"] = [9999, 10]

    def bad_class(meta):
        meta["objects"][0]["class_id"] = 5

    def bad_depth_intrinsics(meta):
        meta["intrinsics_depth"] = meta["intrinsics_image"]

    for i, edit in enumerate([swap_axes, tap_outside, bad_class, bad_depth_intrinsics]):
        report = validate_session(_rewrite(fake_zip, tmp_path / f"bad{i}.zip", edit_meta=edit))
        assert list(report.errors) == ["0001"], (edit.__name__, report.format())
        assert report.n_valid_samples == report.n_samples - 1

    report = validate_session(_rewrite(fake_zip, tmp_path / "nodepth.zip", drop="samples/0002/depth_mm.png"))
    assert list(report.errors) == ["0002"]
    assert main([str(tmp_path / "nodepth.zip")]) == 1
    assert not validate_session(tmp_path / "does_not_exist.zip").ok
