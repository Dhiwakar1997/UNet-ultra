"""Round trip of the fake session through dataset -> model -> metrics, plus reports and export."""

import numpy as np
import pytest
import torch

from unetultra.calibration import fit_from_sessions
from unetultra.capture_report import coverage_by_distance, depth_edge_overlay, format_report, object_coverage
from unetultra.dataset import SessionDataset
from unetultra.metrics import correlation, distance_error_by_bin, evaluate_model, evaluate_sample, summarise
from unetultra.model import UNetMulti
from unetultra.session import Session
from unetultra.train import TrainConfig, fit, load_model, mask_loss, split_sessions

SMALL = 128


@pytest.fixture(scope="module")
def calibration(fake_zips):
    return fit_from_sessions(fake_zips["train"])[0]


def _perfect(item):
    """A prediction equal to the truth: the reviewed labels and the LiDAR depth."""
    return item["labels"].clone(), item["depth"].clone()


def test_perfect_predictions_score_perfectly(fake_zips, calibration):
    dataset = SessionDataset([fake_zips["test"]], calibration)
    object_rows, image_rows = [], []
    for index in range(len(dataset)):
        item = dataset[index]
        rows, image_row = evaluate_sample(*_perfect(item), item, dataset.sample_meta(index))
        object_rows += rows
        image_rows.append(image_row)
    summary = summarise(object_rows, image_rows)
    assert summary["dice"] == 1.0 and summary["missed_fraction"] == 0.0
    assert summary["distance"]["mae"] < 0.05
    assert summary["background_false_positive_fraction"] == 0.0 and summary["n_background_only"] >= 1
    # The bias-fixed size agrees with the ruler truth far better than the reference method.
    assert summary["width_vs_measured"]["mape"] < 4 and summary["height_vs_measured"]["mape"] < 4
    assert summary["width_vs_measured_raw"]["mape"] > summary["width_vs_measured"]["mape"]
    # The calibrated focus distance is a sensible baseline for the focused object.
    assert summary["distance_focus_baseline"]["mape"] < 10
    assert distance_error_by_bin(object_rows) and np.isfinite(correlation(object_rows, "lens_position", "true_distance_cm"))


def test_missed_objects_and_false_positives_are_reported(fake_zips, calibration):
    dataset = SessionDataset([fake_zips["test"]], calibration)
    with_object = next(i for i in range(len(dataset)) if dataset[i]["instances"].max() > 0)
    item = dataset[with_object]
    rows, _ = evaluate_sample(torch.zeros_like(item["labels"]), item["depth"], item)
    assert all(row["missed"] and row["dice"] == 0 and np.isnan(row["pred_distance_cm"]) for row in rows)

    background = next(i for i in range(len(dataset)) if dataset[i]["instances"].max() == 0)
    item = dataset[background]
    everything = torch.ones_like(item["labels"])
    rows, image_row = evaluate_sample(everything, item["depth"], item)
    assert rows == [] and image_row["background_only"] and image_row["predicted_fraction"] == 1.0


def test_training_round_trip(fake_zips, calibration, tmp_path):
    train_paths, val_paths = split_sessions(fake_zips["train"])
    train = SessionDataset(train_paths, calibration, augment=True, size=SMALL)
    val = SessionDataset(val_paths, calibration, size=SMALL)
    config = TrainConfig(epochs=4, batch_size=4, features=(8, 16), lr=3e-3)
    history, best = fit(train, val, config, tmp_path, name="tiny", device="cpu", log=lambda *_: None)
    assert len(history) == 4 and history[-1]["train_loss"] < history[0]["train_loss"]
    assert (tmp_path / "tiny_history.json").exists()

    model, checkpoint = load_model(best)
    assert checkpoint["classes"] == ["background", "airpods_case"] and model.n_classes == 1
    test = SessionDataset([fake_zips["test"]], calibration, size=SMALL)
    object_rows, image_rows, predictions = evaluate_model(model, test)
    summary = summarise(object_rows, image_rows)
    assert len(image_rows) == len(test) == len(predictions)
    assert summary["n_objects"] == len(object_rows) > 0
    assert {"dice", "iou", "distance", "width_vs_lidar", "distance_focus_baseline"} <= set(summary)


def test_multi_class_loss_and_shapes():
    model = UNetMulti(features=(8, 16), n_classes=3)
    logits, _ = model(torch.rand(2, 3, 32, 48), torch.zeros(2, 2))
    labels = torch.randint(0, 4, (2, 32, 48))
    labels[0, :4] = 255  # ignored pixels must not break the loss
    loss = mask_loss(logits, labels, 3)
    assert logits.shape[1] == 4 and torch.isfinite(loss)
    loss.backward()


def test_capture_report(fake_zip):
    with Session(fake_zip) as sess:
        rows = [row for sample_id in sess.sample_ids for row in object_coverage(sess, sample_id)]
        with_object = next(s for s in sess.sample_ids if sess.meta(s)["objects"])
        picture = depth_edge_overlay(sess, with_object)
        assert picture.shape == sess.image(with_object).shape
        assert (picture == (255, 0, 0)).all(axis=-1).sum() > 20  # depth edges were drawn
        assert (picture == (255, 255, 255)).all(axis=-1).sum() > 500  # the crop square was drawn
    assert rows and all(row["coverage"] > 0.8 for row in rows)
    text = format_report(rows, coverage_by_distance(rows))
    assert "Safe minimum distance" in text


def test_coreml_conversion(tmp_path):
    pytest.importorskip("coremltools")
    from unetultra.export import export_coreml

    mlmodel = export_coreml(UNetMulti(features=(8, 16)), ["background", "airpods_case"], tmp_path / "m.mlpackage")
    spec = mlmodel.get_spec()
    assert [i.name for i in spec.description.input] == ["image", "meta"]
    assert [o.name for o in spec.description.output] == ["mask_logits", "depth"]
    assert (tmp_path / "m.mlpackage").is_dir()


def test_single_session_fallback_holds_out_the_last_block(fake_zips, calibration, tmp_path):
    from torch.utils.data import Subset
    from unetultra.train import split_single_session

    one = fake_zips["train"][:1]
    train_indices, val_indices = split_single_session(SessionDataset(one, calibration, size=SMALL))
    full = SessionDataset(one, calibration, size=SMALL)
    is_background = [full.sample_meta(i)["background_only"] for i in range(len(full))]
    objects = [i for i in range(len(full)) if not is_background[i]]
    assert sorted(train_indices + val_indices) == list(range(len(full))) and not set(train_indices) & set(val_indices)
    # The held-out object samples are the last ones captured, as a block, never shuffled.
    held_objects = [i for i in val_indices if not is_background[i]]
    assert held_objects == objects[-len(held_objects):] and len(held_objects) >= 2
    # Training keeps a no-object sample whenever the session has at least two.
    if sum(is_background) >= 2:
        assert any(is_background[i] for i in train_indices) and any(is_background[i] for i in val_indices)
    train = Subset(SessionDataset(one, calibration, augment=True, size=SMALL), train_indices)
    val = Subset(SessionDataset(one, calibration, size=SMALL), val_indices)
    history, best = fit(train, val, TrainConfig(epochs=1, batch_size=4, features=(8, 16)), tmp_path, name="one",
                        device="cpu", log=lambda *_: None)
    assert len(history) == 1 and load_model(best)[1]["classes"] == ["background", "airpods_case"]
    with pytest.raises(ValueError, match="at least 8"):
        split_single_session(list(range(5)))


def test_batches_move_to_a_device_without_float64(fake_zips, calibration):
    """MPS has no float64; the collated focus distance and lens position must be converted."""
    from torch.utils.data import DataLoader
    from unetultra.train import _to_device

    batch = next(iter(DataLoader(SessionDataset(fake_zips["train"][:1], calibration, size=SMALL), batch_size=4)))
    assert batch["focus_distance_m"].dtype == torch.float64  # the trap
    moved = _to_device(batch, "cpu")
    assert all(v.dtype != torch.float64 for v in moved.values() if torch.is_tensor(v))
    assert moved["sample_id"] == batch["sample_id"]


def test_checkpoint_choice_ignores_a_lucky_early_epoch():
    """A noisy early epoch with a low distance error but poor masks must not stay the "best"."""
    from unetultra.train import checkpoint_score, is_new_best

    def epoch(dice, error):
        return {"val_dice": dice, "val_distance_error_cm": error, "val_loss": 1.0}

    curve = [epoch(0.10, 30), epoch(0.30, 25), epoch(0.50, 6), epoch(0.70, 22), epoch(0.92, 12),
             epoch(0.93, 11), epoch(0.93, 10), epoch(0.94, 13), epoch(0.90, 4)]
    history, saved, kept = [], None, []
    for number, row in enumerate(curve, start=1):
        history.append(row)
        if is_new_best(history, saved):
            saved = {"score": checkpoint_score(history), "dice": row["val_dice"]}
            kept.append(number)
    assert 3 in kept            # the lucky epoch is taken at first, while it is the best known
    assert kept[-1] == 7        # but well-trained epochs replace it; the best smoothed error wins
    assert 9 not in kept        # a final fluke with clearly worse masks is ignored
    assert checkpoint_score(history[:7]) == pytest.approx(11.0)
