"""Model registry, the DINOv2-integrated model, and export as .unetmodel bundles."""

import io
import json
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from unetultra.calibration import fit_from_sessions
from unetultra.dataset import SessionDataset, meta_from_features
from unetultra.models import ARCHITECTURES, build_model, meta_features, uses_meta
from unetultra.train import TrainConfig, fit, load_model, split_sessions

SMALL = 128
# An untrained DINOv2 (no download) on a small input, so the tests stay fast.
EMBED_OPTIONS = {"pretrained": False, "dino_input": 112}


@pytest.fixture(scope="module")
def calibration(fake_zips):
    return fit_from_sessions(fake_zips["train"])[0]


def test_registry():
    assert set(ARCHITECTURES) >= {"unet_multi", "image_only", "unet_embed"}
    assert meta_features("unet_multi") == ["log_focus_distance_m", "fx_norm"] == meta_features("unet_embed")
    assert meta_features("image_only") == [] and not uses_meta("image_only") and uses_meta("unet_embed")
    with pytest.raises(ValueError, match="unknown architecture"):
        build_model("nope")
    assert build_model("image_only", features=(8, 16)).architecture == "image_only"


def test_meta_from_features():
    values = meta_from_features(["fx_norm", "log_focus_distance_m", "lens_position_raw"], 0.5, 400.0, 0.3)
    assert values.tolist() == pytest.approx([-0.6, np.log(0.5), 0.3])
    assert meta_from_features([], 0.5, 400.0, 0.3).shape == (0,)
    with pytest.raises(ValueError, match="unknown meta feature"):
        meta_from_features(["zoom"], 0.5, 400.0, 0.3)


def test_unet_embed_shapes_frozen_encoder_and_prototype(fake_zips, calibration):
    model = build_model("unet_embed", features=(8, 16), **EMBED_OPTIONS)
    assert all(not p.requires_grad for p in model.dino.parameters())
    assert sum(p.requires_grad for p in model.parameters()) > 0
    model.train()
    assert not model.dino.training  # the encoder stays in eval mode
    mask, depth = model(torch.rand(2, 3, SMALL, SMALL), torch.zeros(2, 2))
    assert mask.shape == (2, 1, SMALL, SMALL) and depth.shape == (2, 1, SMALL, SMALL) and (depth > 0).all()
    assert model.dino(torch.rand(1, 3, SMALL, SMALL)).shape == (1, 384, 8, 8)

    assert not model.prototype.any()
    used = model.compute_prototype(SessionDataset(fake_zips["train"][:1], calibration, size=SMALL))
    assert used >= 5 and float(model.prototype.norm()) == pytest.approx(1.0, abs=1e-4)


def test_frozen_dino_matches_the_library_forward():
    """Our fixed-size forward (constant position embeddings) equals DINOv2's own."""
    model = build_model("unet_embed", features=(8, 16), **EMBED_OPTIONS).eval()
    image = torch.rand(1, 3, 112, 112)
    with torch.no_grad():
        ours = model.dino(image).flatten(2).transpose(1, 2)
        normalised = (image - model.dino.mean) / model.dino.std
        reference = model.dino.backbone(pixel_values=normalised).last_hidden_state[:, 1:, :]
    assert torch.allclose(ours, reference, atol=1e-5)


@pytest.mark.parametrize("architecture", ["unet_multi", "image_only", "unet_embed"])
def test_every_architecture_trains_saves_and_reloads(architecture, fake_zips, calibration, tmp_path):
    train_paths, val_paths = split_sessions(fake_zips["train"])
    use_meta = uses_meta(architecture)
    train = SessionDataset(train_paths, calibration, augment=True, size=SMALL, use_meta=use_meta)
    val = SessionDataset(val_paths, calibration, size=SMALL, use_meta=use_meta)
    config = TrainConfig(architecture=architecture, epochs=1, batch_size=4, features=(8, 16),
                         options=EMBED_OPTIONS if architecture == "unet_embed" else None)
    _, best = fit(train, val, config, tmp_path, name=architecture, device="cpu", log=lambda *_: None)
    model, checkpoint = load_model(best)
    assert checkpoint["architecture"] == architecture == model.architecture
    assert checkpoint["sessions"] == sorted(p.split("/")[-1] for p in train_paths)
    if architecture == "unet_embed":
        assert float(model.prototype.norm()) == pytest.approx(1.0, abs=1e-4)  # prototype travels with the checkpoint
    item = val[0]
    with torch.no_grad():
        mask, depth = model(item["image"][None], item["meta"][None])
    assert mask.shape[-2:] == depth.shape[-2:] == (SMALL, SMALL)


def test_old_checkpoints_without_an_architecture_still_load(tmp_path):
    model = build_model("unet_multi", features=(8, 16))
    for use_meta, expected in ((True, "unet_multi"), (False, "image_only")):
        torch.save({"state_dict": model.state_dict(), "n_classes": 1, "classes": ["background", "x"], "epoch": 1,
                    "config": {"features": (8, 16), "use_meta": use_meta}}, tmp_path / "old.pt")
        assert load_model(tmp_path / "old.pt")[1]["architecture"] == expected


@pytest.mark.parametrize("architecture", ["unet_multi", "image_only"])
def test_export_bundle(architecture, fake_zips, calibration, tmp_path):
    pytest.importorskip("coremltools")
    from unetultra import export
    from unetultra.metrics import evaluate_predictor, summarise

    model = build_model(architecture, features=(8, 16))
    torch.save({"state_dict": model.state_dict(), "n_classes": 1, "classes": ["background", "airpods_case"],
                "epoch": 1, "architecture": architecture, "sessions": ["a.zip"],
                "config": {"features": (8, 16), "use_meta": uses_meta(architecture)}}, tmp_path / "m.pt")
    dataset = SessionDataset([fake_zips["test"]], calibration, use_meta=uses_meta(architecture))
    bundle, card, _ = export.export_bundle(tmp_path / "m.pt", f"test-{architecture}", tmp_path / "exports", dataset,
                                           precision="fp16", display_name="Test", test_metrics={"dice": 0.9},
                                           log=lambda *_: None)
    assert bundle.name == f"test-{architecture}.unetmodel"
    assert card["contract_version"] == 1 and card["architecture"] == architecture
    assert card["input"] == {"width": 512, "height": 512, "crop": "center_square"}
    assert card["meta_features"] == meta_features(architecture)
    assert card["test_metrics"] == {"dice": 0.9, "distance_err_pct": None, "size_err_pct": None}
    assert card["trained_on_sessions"] == ["a.zip"] and card["size_mb"] > 0 and len(card["sha256"]) == 64

    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
        assert all(info.compress_type == zipfile.ZIP_STORED for info in zf.infolist())  # the phone reads stored only
        assert "model_card.json" in names and any(n.startswith("model.mlpackage/") for n in names)
        assert export.read_card(bundle) == json.loads(zf.read("model_card.json")) == card
        picture = Image.open(io.BytesIO(zf.read("self_test/input.jpg")))
        assert picture.size == (512, 512)
        meta = json.loads(zf.read("self_test/meta.json"))
        assert meta["meta_features"] == card["meta_features"] and len(meta["values"]) == len(card["meta_features"])
        mask = np.array(Image.open(io.BytesIO(zf.read("self_test/expected_mask.png"))))
        depth = np.load(io.BytesIO(zf.read("self_test/expected_depth.npy")))
        assert mask.shape == depth.shape == (512, 512) and mask.dtype == np.uint8 and depth.dtype == np.float32
        # The expected outputs are exactly what PyTorch gives for the decoded JPEG.
        logits, expected = export.torch_predict(model, architecture, np.array(picture.convert("RGB")), meta["values"])
        assert np.allclose(depth, expected[0, 0], atol=1e-5) and (mask == export.class_map(logits)).all()

    # The Core ML model has a meta input only when the architecture uses one.
    import coremltools as ct
    package = export.extract_package(bundle, tmp_path / "unpacked")
    spec = ct.utils.load_spec(str(package))
    assert [i.name for i in spec.description.input] == (["image", "meta"] if uses_meta(architecture) else ["image"])
    assert [o.name for o in spec.description.output] == ["mask_logits", "depth"]

    # Any predictor with the bundle's signature can be scored (here PyTorch stands in for Core ML).
    rows, image_rows, latencies = evaluate_predictor(
        lambda image, meta: export.torch_predict(model, architecture, image, meta), dataset, card["meta_features"])
    assert len(image_rows) == len(dataset) == len(latencies) and "dice" in summarise(rows, image_rows)
    assert set(export.headline_metrics(summarise(rows, image_rows))) == {"dice", "distance_err_pct", "size_err_pct"}
