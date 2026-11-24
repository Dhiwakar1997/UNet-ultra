"""Export a trained checkpoint as a model bundle for the phone (see docs/model_contract.md).

    python -m unetultra.export --checkpoint models/model.pt --name unet-multi-512 --out exports/

A bundle `<name>.unetmodel` is a zip holding the Core ML model, a model card and a self-test.
Every bundle has the same outside shape, whatever is inside it:

    inputs   image (the centre square of the frame, already cropped and resized by the caller;
             scaling and any normalisation are inside the model), and optionally meta [1, N]
    outputs  mask_logits [1, C, H, W], and optionally depth [1, 1, H, W] in metres

Conversion works anywhere; running the converted model (the PyTorch comparison) needs macOS.
"""

import argparse
import hashlib
import io
import json
import platform
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .model import INPUT_HEIGHT, INPUT_WIDTH
from .models import meta_features

CONTRACT_VERSION = 1
IMAGE_INPUT, META_INPUT = "image", "meta"
MASK_OUTPUT, DEPTH_OUTPUT = "mask_logits", "depth"
BUNDLE_SUFFIX = ".unetmodel"
MODEL_ENTRY, CARD_ENTRY, SELF_TEST = "model.mlpackage", "model_card.json", "self_test"


class ContractModel(nn.Module):
    """Wraps a trained model in the contract's signature: (image, meta?) -> (mask_logits, depth).

    The image arrives in 0..1 (Core ML applies the 1/255 scale). A model that takes no meta
    features is given zeros internally, so its Core ML form has no `meta` input at all."""

    def __init__(self, model, features):
        super().__init__()
        self.model = model
        self.features = list(features)
        self.register_buffer("no_meta", torch.zeros(1, 2))  # what an image-only model was trained with

    def forward(self, image, meta=None):
        """One image at a time, as on the phone."""
        if not self.features:
            meta = self.no_meta
        return self.model(image, meta)


def contract_model(model, architecture):
    return ContractModel(model.cpu().eval(), meta_features(architecture)).eval()


def trace_model(wrapped):
    """TorchScript trace at the fixed export size, with inputs in the training range (image 0..1)."""
    image = torch.rand(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    example = (image, torch.zeros(1, len(wrapped.features))) if wrapped.features else (image,)
    return torch.jit.trace(wrapped, example, check_trace=False)


def export_coreml(model, classes, out_path, architecture="unet_multi", precision="fp32",
                  description="UNet-ultra: mask and depth from RGB + focus"):
    """Convert a trained model to an ML Program .mlpackage for iOS 17+. Returns the MLModel."""
    import coremltools as ct

    wrapped = contract_model(model, architecture)
    inputs = [ct.ImageType(name=IMAGE_INPUT, shape=(1, 3, INPUT_HEIGHT, INPUT_WIDTH), scale=1 / 255.0,
                           color_layout=ct.colorlayout.RGB)]
    if wrapped.features:
        inputs.append(ct.TensorType(name=META_INPUT, shape=(1, len(wrapped.features)), dtype=np.float32))
    mlmodel = ct.convert(
        trace_model(wrapped),
        inputs=inputs,
        outputs=[ct.TensorType(name=MASK_OUTPUT), ct.TensorType(name=DEPTH_OUTPUT)],
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32,
    )
    mlmodel.short_description = description
    mlmodel.input_description[IMAGE_INPUT] = "Centre square of the RGB frame resized to 512x512, sensor orientation"
    if wrapped.features:
        mlmodel.input_description[META_INPUT] = "meta features, in order: " + ", ".join(wrapped.features)
    mlmodel.output_description[MASK_OUTPUT] = "mask logits: 1 channel, or classes + background"
    mlmodel.output_description[DEPTH_OUTPUT] = "per-pixel depth in metres"
    mlmodel.user_defined_metadata["classes"] = json.dumps(list(classes))
    mlmodel.user_defined_metadata["n_classes"] = str(model.n_classes)
    mlmodel.user_defined_metadata["crop"] = "center_square"
    mlmodel.user_defined_metadata["meta_features"] = json.dumps(wrapped.features)
    out_path = Path(out_path)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    return mlmodel


def class_map(mask_logits, threshold=0.5):
    """Logits [1, C, H, W] (numpy) -> class id per pixel, uint8 [H, W]."""
    if mask_logits.shape[1] == 1:
        return (mask_logits[0, 0] > np.log(threshold / (1 - threshold))).astype(np.uint8)
    return mask_logits[0].argmax(0).astype(np.uint8)


def torch_predict(model, architecture, image_u8, meta):
    """Run the PyTorch model exactly as the bundle will be run: 8-bit image in, numpy out."""
    wrapped = contract_model(model, architecture)
    image = torch.from_numpy(np.array(image_u8)).permute(2, 0, 1)[None].float() / 255.0
    with torch.no_grad():
        if wrapped.features:
            logits, depth = wrapped(image, torch.from_numpy(np.asarray(meta, dtype=np.float32))[None])
        else:
            logits, depth = wrapped(image)
    return logits.numpy(), depth.numpy()


def coreml_predict(mlmodel, image_u8, meta):
    """Run a converted model on one 8-bit image (macOS only). Returns (mask_logits, depth)."""
    inputs = {IMAGE_INPUT: Image.fromarray(np.asarray(image_u8))}
    if len(meta):
        inputs[META_INPUT] = np.asarray(meta, dtype=np.float32)[None]
    out = mlmodel.predict(inputs)
    return np.asarray(out[MASK_OUTPUT]), np.asarray(out[DEPTH_OUTPUT])


def item_meta(item, features):
    """The meta values for one dataset item, for a list of named features."""
    from .dataset import meta_from_features

    return meta_from_features(features, float(item["focus_distance_m"]), float(item["intrinsics"][0]),
                              float(item["lens_position"]))


def compare_with_pytorch(mlmodel, model, dataset, indices, architecture="unet_multi"):
    """Run both models on real samples and return per-sample maximum absolute differences.

    Both get the same 8-bit image, as on the phone. Needs macOS."""
    features = meta_features(architecture)
    rows = []
    for index in indices:
        item = dataset[index]
        image_u8 = (item["image"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        meta = item_meta(item, features)
        ml_logits, ml_depth = coreml_predict(mlmodel, image_u8, meta)
        pt_logits, pt_depth = torch_predict(model, architecture, image_u8, meta)
        rows.append({
            "sample_id": item["sample_id"],
            "mask_logits_max_abs_diff": float(np.abs(ml_logits - pt_logits).max()),
            "depth_max_abs_diff_m": float(np.abs(ml_depth - pt_depth).max()),
            "mask_pixels_changed": int((class_map(ml_logits) != class_map(pt_logits)).sum()),
        })
    return rows


# ---- bundles --------------------------------------------------------------------------------

def _npy_bytes(array):
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array, dtype="<f4"), allow_pickle=False)
    return buffer.getvalue()


def _jpeg_roundtrip(image_u8):
    """Encode as JPEG and decode again, so the expected outputs match the pixels the phone will decode."""
    buffer = io.BytesIO()
    Image.fromarray(image_u8).save(buffer, format="JPEG", quality=95)
    data = buffer.getvalue()
    return data, np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def _package_sha256(package):
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(package).rglob("*") if p.is_file()):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_bundle(out_path, package, card, self_test_files):
    """Zip a .mlpackage, its card and the self-test files into a .unetmodel.

    Entries are stored, not compressed: the weights barely compress, and the phone's reader
    then needs no decompression code."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as zf:
        for path in sorted(p for p in Path(package).rglob("*") if p.is_file()):
            zf.write(path, f"{MODEL_ENTRY}/{path.relative_to(package).as_posix()}")
        zf.writestr(CARD_ENTRY, json.dumps(card, indent=2))
        for name, data in self_test_files.items():
            zf.writestr(f"{SELF_TEST}/{name}", data)
    return out_path


def read_card(bundle_path):
    with zipfile.ZipFile(bundle_path) as zf:
        return json.loads(zf.read(CARD_ENTRY))


def extract_package(bundle_path, directory):
    """Unpack a bundle's Core ML package into `directory` and return the .mlpackage path."""
    with zipfile.ZipFile(bundle_path) as zf:
        zf.extractall(directory, [n for n in zf.namelist() if n.startswith(MODEL_ENTRY + "/")])
    return Path(directory) / MODEL_ENTRY


def export_bundle(checkpoint_path, name, out_dir, dataset=None, precision="fp16", display_name=None,
                  test_metrics=None, notes="", compare_samples=6, log=print):
    """Checkpoint -> `<out_dir>/<name>.unetmodel`. Returns (bundle_path, card, comparison_rows).

    `dataset` supplies the real samples for the self-test and for the Core ML vs PyTorch check;
    without one, a synthetic image is used for the self-test and the check is skipped."""
    from .train import load_model

    model, checkpoint = load_model(checkpoint_path)
    architecture, classes = checkpoint["architecture"], checkpoint["classes"]
    features = meta_features(architecture)

    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp) / MODEL_ENTRY
        log(f"converting {architecture} to Core ML ({precision})")
        mlmodel = export_coreml(model, classes, package, architecture, precision)

        comparison = []
        if dataset is not None and len(dataset) and platform.system() == "Darwin":
            indices = list(range(0, len(dataset), max(1, len(dataset) // compare_samples)))[:compare_samples]
            comparison = compare_with_pytorch(mlmodel, model, dataset, indices, architecture)
            log(f"Core ML vs PyTorch on {len(comparison)} samples: "
                f"mask logits max diff {max(r['mask_logits_max_abs_diff'] for r in comparison):.3g}, "
                f"depth max diff {max(r['depth_max_abs_diff_m'] for r in comparison) * 1000:.2f} mm, "
                f"mask pixels changed {max(r['mask_pixels_changed'] for r in comparison)}")
        elif platform.system() != "Darwin":
            log("Core ML vs PyTorch check skipped: Core ML models only run on macOS")

        # Self-test: one real image (with an object if there is one) and the PyTorch outputs for it.
        if dataset is not None and len(dataset):
            index = next((i for i in range(len(dataset)) if int(dataset[i]["instances"].max()) > 0), 0)
            item = dataset[index]
            image_u8 = (item["image"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            meta = item_meta(item, features)
        else:
            image_u8 = np.random.default_rng(0).integers(0, 255, (INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
            meta = np.array([np.log(0.4), -0.5][: len(features)], dtype=np.float32)
        jpeg, decoded = _jpeg_roundtrip(image_u8)
        logits, depth = torch_predict(model, architecture, decoded, meta)
        mask_png = io.BytesIO()
        Image.fromarray(class_map(logits)).save(mask_png, format="PNG")
        self_test = {
            "input.jpg": jpeg,
            "meta.json": json.dumps({"meta_features": features, "values": [float(v) for v in meta]}),
            "expected_mask.png": mask_png.getvalue(),
            "expected_depth.npy": _npy_bytes(depth[0, 0]),
        }

        size_mb = sum(p.stat().st_size for p in package.rglob("*") if p.is_file()) / 1e6
        card = {
            "contract_version": CONTRACT_VERSION,
            "name": name,
            "display_name": display_name or name,
            "version": datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M"),
            "architecture": architecture,
            "input": {"width": INPUT_WIDTH, "height": INPUT_HEIGHT, "crop": "center_square"},
            "meta_features": features,
            "outputs": {MASK_OUTPUT: {"channels": int(logits.shape[1])}, DEPTH_OUTPUT: {"units": "m"}},
            "classes": list(classes),
            "mask_threshold": 0.5,
            "trained_on_sessions": checkpoint.get("sessions", []),
            "test_metrics": {"dice": None, "distance_err_pct": None, "size_err_pct": None, **(test_metrics or {})},
            "size_mb": round(size_mb, 2),
            "precision": precision,
            "notes": notes,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": _package_sha256(package),
        }
        bundle = write_bundle(Path(out_dir) / f"{name}{BUNDLE_SUFFIX}", package, card, self_test)
    log(f"wrote {bundle} ({bundle.stat().st_size / 1e6:.1f} MB)")
    return bundle, card, comparison


def headline_metrics(summary):
    """The three card metrics from a metrics.summarise() result."""
    def value(key, field):
        number = summary.get(key, {}).get(field)
        return None if number is None or not np.isfinite(number) else round(float(number), 2)

    size = [v for v in (value("width_vs_lidar", "mape"), value("height_vs_lidar", "mape")) if v is not None]
    dice = summary.get("dice")
    return {"dice": None if dice is None else round(float(dice), 3),
            "distance_err_pct": value("distance", "mape"),
            "size_err_pct": round(float(np.mean(size)), 2) if size else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export a checkpoint as a .unetmodel bundle for the phone.")
    parser.add_argument("--checkpoint", required=True, help="checkpoint written by training, e.g. models/model.pt")
    parser.add_argument("--name", required=True, help="bundle name, e.g. unet-multi-512")
    parser.add_argument("--out", default="exports", help="output directory")
    parser.add_argument("--display-name")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--data", default="data", help="folder of session zips, for the self-test and the check")
    parser.add_argument("--calibration", default="models/calibration.json")
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)

    from .calibration import LensCalibration
    from .dataset import SessionDataset
    from .session import find_sessions

    dataset, metrics = None, None
    zips = find_sessions(args.data, "test") or find_sessions(args.data) if Path(args.data).is_dir() else []
    if zips and Path(args.calibration).exists():
        dataset = SessionDataset(zips, LensCalibration.load(args.calibration))
    else:
        print("no sessions or calibration found: the self-test will use a synthetic image")
    metrics_file = Path(args.checkpoint).with_name(Path(args.checkpoint).stem + "_test_metrics.json")
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())
    _, card, comparison = export_bundle(args.checkpoint, args.name, args.out, dataset, args.precision,
                                        args.display_name, metrics, args.notes)
    if comparison:
        worst_depth = max(row["depth_max_abs_diff_m"] for row in comparison)
        tolerance = 0.01 if args.precision == "fp16" else 0.001
        print("PASS" if worst_depth < tolerance else f"WARNING: depth differs by {worst_depth * 1000:.1f} mm")
    print(json.dumps({k: card[k] for k in ("name", "architecture", "meta_features", "size_mb", "precision")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
