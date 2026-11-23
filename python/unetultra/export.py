"""Core ML export and the PyTorch/Core ML comparison (used by 04_export_coreml.ipynb).

The Core ML model takes `image` (512x512 RGB, the centre square of the frame already cropped
and resized by the caller; the 1/255 scaling is built in) and `meta`
([1, 2] float32) and returns `mask_logits` and `depth` (metres). Conversion works anywhere;
running the converted model for the comparison needs macOS.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import INPUT_HEIGHT, INPUT_WIDTH

IMAGE_INPUT, META_INPUT = "image", "meta"
MASK_OUTPUT, DEPTH_OUTPUT = "mask_logits", "depth"


def trace_model(model):
    """TorchScript trace at the fixed export size, with inputs in the training range (image 0..1)."""
    model = model.cpu().eval()
    example = (torch.rand(1, 3, INPUT_HEIGHT, INPUT_WIDTH), torch.zeros(1, 2))
    return torch.jit.trace(model, example, check_trace=False)


def export_coreml(model, classes, out_path, description="UNet-ultra: mask and depth from RGB + focus"):
    """Convert a trained UNetMulti to an ML Program .mlpackage for iOS 17+. Returns the MLModel."""
    import coremltools as ct

    mlmodel = ct.convert(
        trace_model(model),
        inputs=[
            ct.ImageType(name=IMAGE_INPUT, shape=(1, 3, INPUT_HEIGHT, INPUT_WIDTH), scale=1 / 255.0,
                         color_layout=ct.colorlayout.RGB),
            ct.TensorType(name=META_INPUT, shape=(1, 2), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name=MASK_OUTPUT), ct.TensorType(name=DEPTH_OUTPUT)],
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
    )
    mlmodel.short_description = description
    mlmodel.input_description[IMAGE_INPUT] = "Centre square of the RGB frame resized to 512x512, sensor orientation"
    mlmodel.input_description[META_INPUT] = "[log(calibrated focus distance in m), fx of the 512x512 crop / 1000 - 1]"
    mlmodel.output_description[MASK_OUTPUT] = "mask logits: 1 channel, or classes + background"
    mlmodel.output_description[DEPTH_OUTPUT] = "per-pixel depth in metres"
    mlmodel.user_defined_metadata["classes"] = json.dumps(list(classes))
    mlmodel.user_defined_metadata["n_classes"] = str(model.n_classes)
    mlmodel.user_defined_metadata["crop"] = "center_square"
    out_path = Path(out_path)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    return mlmodel


def compare_with_pytorch(mlmodel, model, dataset, indices):
    """Run both models on real samples and return per-sample maximum absolute differences.

    The Core ML model gets the 8-bit image, as it will on the phone; PyTorch gets the same
    pixels as floats. Needs macOS."""
    model = model.cpu().eval()
    rows = []
    for index in indices:
        item = dataset[index]
        image_u8 = (item["image"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        meta = item["meta"].numpy()[None].astype(np.float32)
        out = mlmodel.predict({IMAGE_INPUT: Image.fromarray(image_u8), META_INPUT: meta})
        with torch.no_grad():
            image = torch.from_numpy(image_u8).permute(2, 0, 1)[None].float() / 255.0
            mask_logits, depth = model(image, torch.from_numpy(meta))
        rows.append({
            "sample_id": item["sample_id"],
            "mask_logits_max_abs_diff": float(np.abs(out[MASK_OUTPUT] - mask_logits.numpy()).max()),
            "depth_max_abs_diff_m": float(np.abs(out[DEPTH_OUTPUT] - depth.numpy()).max()),
            "mask_pixels_changed": int(((out[MASK_OUTPUT] > 0) != (mask_logits.numpy() > 0)).sum())
            if model.n_classes == 1 else
            int((out[MASK_OUTPUT].argmax(1) != mask_logits.numpy().argmax(1)).sum()),
        })
    return rows
