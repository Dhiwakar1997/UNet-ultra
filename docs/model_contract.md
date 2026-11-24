# Model contract

`contract_version: 1`

Every model the app can run has the same outside shape, whatever is inside it. A model that
follows this contract can be sent to the phone and used without rebuilding the app. The app
refuses a bundle whose `contract_version` has a major version it does not know, and says why.

## Inputs

| Name | Type | Required | Meaning |
| --- | --- | --- | --- |
| `image` | RGB image, size given in the model card | yes | The centre square of the camera frame, resized to the model input. Scaling and normalisation happen inside the model, so the caller only crops and resizes. |
| `meta` | float32 `[1, N]` | no | The features named in the card's `meta_features`, in that order. An image-only model has no `meta` input. |

The meta features the app knows how to compute:

| Feature | Value |
| --- | --- |
| `log_focus_distance_m` | natural log of the calibrated focus distance in metres |
| `fx_norm` | fx of the cropped and resized image, divided by 1000, minus 1 |
| `lens_position_raw` | the raw lens reading, 0 to 1 (for experiments only) |

A card that names any other feature cannot be run.

## Outputs

| Name | Shape | Required | Meaning |
| --- | --- | --- | --- |
| `mask_logits` | `[1, C, H, W]` | yes | C = 1: sigmoid, object where the probability exceeds `mask_threshold`. C > 1: softmax, class 0 is background. |
| `depth` | `[1, 1, H, W]` | no | Per-pixel depth in metres. Without it, the app uses the calibrated focus distance as the object's distance. |

Anything model-specific stays inside the Core ML model. For `unet_embed`, the frozen DINOv2
encoder, its ImageNet normalisation and the class prototype are all part of the exported
model, so from outside it is indistinguishable from `unet_multi`.

## Pre-processing: the crop

`input.crop` is `"center_square"`, the only supported value. For a frame of width W and height H:

```
S  = min(W, H)        x0 = (W - S) / 2        y0 = (H - S) / 2        s = input.width / S

fx' = fx * s                         fy' = fy * s
cx' = (cx - x0 + 0.5) * s - 0.5      cy' = (cy - y0 + 0.5) * s - 0.5
```

Points transform the same way as the principal point. The frame is never rotated: it stays in
sensor orientation. The single implementation of this is `geometry.CenterSquareCrop` in Python
and `CenterSquareCrop` in the app.

## The bundle: `<name>.unetmodel`

A zip with a custom extension. Entries are **stored, not compressed**.

```
<name>.unetmodel
  model.mlpackage/           Core ML model following this contract (iOS 17+)
  model_card.json
  self_test/
    input.jpg                one real image, already cropped and resized to the model input
    meta.json                {"meta_features": [...], "values": [...]} used for the self-test
    expected_mask.png        8-bit class id per pixel from PyTorch for this input
    expected_depth.npy       float32 [H, W] depth in metres from PyTorch (absent if no depth output)
```

The expected outputs are computed from the decoded `input.jpg`, so both sides see the same pixels.

## `model_card.json`

```json
{
  "contract_version": 1,
  "name": "unet-embed-dinov2s-512",
  "display_name": "U-Net + DINOv2-S",
  "version": "2026-11-24-1042",
  "architecture": "unet_embed",
  "input": {"width": 512, "height": 512, "crop": "center_square"},
  "meta_features": ["log_focus_distance_m", "fx_norm"],
  "outputs": {"mask_logits": {"channels": 1}, "depth": {"units": "m"}},
  "classes": ["background", "airpods_case"],
  "mask_threshold": 0.5,
  "trained_on_sessions": ["session_....zip"],
  "test_metrics": {"dice": null, "distance_err_pct": null, "size_err_pct": null},
  "size_mb": 50.5,
  "precision": "fp16",
  "notes": "",
  "created_at": "ISO-8601",
  "sha256": "of the model.mlpackage contents"
}
```

- `meta_features` is `[]` for an image-only model.
- `outputs.depth` is absent for a model with no depth output.
- `test_metrics` are copied from `03_evaluate.ipynb`, so laptop and phone numbers can be compared.
- Readers ignore keys they do not know.

## Self-test

On import the app runs `input.jpg` and the `meta.json` values through the compiled model and
compares the result with the expected files. A failure means the phone's pre-processing or the
conversion does not match training; the model is marked with a warning and can be used only
after explicit confirmation.

## Registered architectures

| Name | What it is | Meta features |
| --- | --- | --- |
| `unet_multi` | U-Net with FiLM conditioning on the focus input | `log_focus_distance_m`, `fx_norm` |
| `image_only` | The same U-Net with no focus input (ablation) | none |
| `unet_embed` | U-Net with the focus input and frozen DINOv2-small features fused at the bottleneck | `log_focus_distance_m`, `fx_norm` |

To add one, write a builder in `python/unetultra/models/` and register it in `ARCHITECTURES`.
