import numpy as np
import pytest

from unetultra.calibration import LensCalibration, collect_calibration_points, fit_calibration, fit_from_sessions
from unetultra.fake_session import fake_lens_position


def test_fit_recovers_a_monotonic_curve():
    rng = np.random.default_rng(0)
    distance = rng.uniform(0.15, 0.7, 200)
    lens = np.array([fake_lens_position(d) for d in distance]) + rng.normal(0, 0.003, 200)
    calibration = fit_calibration(lens, distance, "FakePhone")
    assert calibration.rmse_m < 0.02
    assert np.all(np.diff(calibration.lens_position) > 0)
    assert np.all(np.diff(calibration.distance_m) >= 0)
    for d in (0.2, 0.35, 0.6):
        assert calibration.focus_distance_m(fake_lens_position(d)) == pytest.approx(d, rel=0.08)


def test_fit_handles_a_lens_that_runs_the_other_way():
    distance = np.linspace(0.15, 0.7, 50)
    calibration = fit_calibration(0.1 + 0.12 / distance, distance)
    assert calibration.focus_distance_m(0.1 + 0.12 / 0.3) == pytest.approx(0.3, rel=0.03)


def test_out_of_range_readings_are_clamped_and_json_round_trips(tmp_path):
    distance = np.linspace(0.2, 0.6, 30)
    calibration = fit_calibration(1 - 0.125 / distance, distance, "FakePhone")
    assert calibration.focus_distance_m(0.0) == pytest.approx(0.2, rel=0.02)
    assert calibration.focus_distance_m(1.0) == pytest.approx(0.6, rel=0.02)
    loaded = LensCalibration.load(calibration.save(tmp_path / "calibration.json"))
    assert loaded.device_model == "FakePhone" and loaded.n_samples == 30
    assert loaded.focus_distance_m(0.5) == pytest.approx(calibration.focus_distance_m(0.5))


def test_needs_enough_points():
    with pytest.raises(ValueError):
        fit_calibration([0.1, 0.2], [0.2, 0.3])


def test_points_come_from_lidar_at_the_focused_tap(fake_zips):
    points = collect_calibration_points(fake_zips["train"])
    assert len(points["lens_position"]) >= 20  # background-only samples contribute nothing
    assert points["distance_m"].min() > 0.14 and points["distance_m"].max() < 0.71
    calibration, _ = fit_from_sessions(fake_zips["train"])
    assert calibration.rmse_m < 0.03 and calibration.device_model == "FakePhone1,1"
