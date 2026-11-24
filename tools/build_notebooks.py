#!/usr/bin/env python
"""Regenerate the four notebooks from the cell sources below (keeps the .ipynb diffs readable)."""

import json
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
SETUP = """import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "python"))  # use the package without installing it
DATA = Path("../data")      # session zips exported from the phone
MODELS = Path("../models")  # checkpoints, calibration.json, UNetUltra.mlpackage
MODELS.mkdir(exist_ok=True)"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text, tag=None):
    metadata = {"tags": [tag]} if tag else {}
    return {"cell_type": "code", "metadata": metadata, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


def write(name, cells):
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (NOTEBOOKS / f"{name}.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")


write("01_masks", [
    md("""
# 01 · Masks

Turn the tap points saved on the phone into masks with **SAM 2**, then review them one by one.

1. SAM 2.1 runs once per tapped object, with the tap as a positive point. Of SAM's guesses, the
   one whose size best fits a ~5.5 cm object at the measured distance is used.
2. You step through the samples: **accept**, **reject**, or **edit** the mask and then accept.
3. `labels.png`, `instances.png`, your edits and `mask_status` are written back into each zip.

SAM is used only here, to make training labels. It is never used at inference.

Run this notebook in **Jupyter Lab in the browser** (`jupyter lab notebooks`); the click-to-edit
image needs it. It is quickest to make the masks from a terminal first, with
`python -m unetultra.masks data/*.zip` (add `--redo` to start again from scratch): this notebook
then loads them and goes straight to the review.
"""),
    code(SETUP),
    code("""
from unetultra.session import find_sessions
from unetultra.validate import validate_session

ZIPS = find_sessions(DATA)          # or list the zips you want: ["../data/session_....zip"]
for path in ZIPS:
    print(validate_session(path).format())
"""),
    md("## Load SAM 2\n\nNeeded for editing, and for any sample that has no mask yet."),
    code("""
from unetultra.masks import Sam2Predictor

# CPU is the safe choice (about a second per object). device="mps" has been seen to stall.
predictor = Sam2Predictor("facebook/sam2.1-hiera-small", device="cpu")   # or "facebook/sam2.1-hiera-base-plus"
print("SAM 2 running on", predictor.device)
""", tag="predictor"),
    code("""
import time
from unetultra.masks import MaskReview

review = MaskReview(ZIPS, predictor, object_size_cm=5.5)
started = time.time()
# Only samples without a saved mask are run. One line each, printed before SAM starts on it.
review.run(progress=lambda done, total, sample_id: print(
    f"{done}/{total}  sample {sample_id}  ({time.time() - started:.0f}s so far)", flush=True))
print(f"done in {time.time() - started:.0f}s")
review.counts()
"""),
    md("""
## Review one by one

The picture is zoomed on the tapped object; the mask is tinted with a white edge, and the tap is
the yellow cross.

- **accept & next** / **reject & next**: decide and move on. **back** returns to the previous one.
- If the mask is wrong, pick a mode and click on the picture:
  - **add to mask**: click a part of the case that is missing.
  - **remove from mask**: click something wrongly covered (table, hand).
  - **restart from click**: the tap landed beside the case; click the case and SAM starts again there.
  - **draw outline**: click points around the case, then **close outline**. That shape becomes the mask.
- **other SAM guess** switches between SAM's alternative masks for the same clicks.
- **reset mask** undoes every edit; **zoom in/out** shows the whole photo.

Progress is saved every 10 decisions and when you reach the end, so you can stop and come back:
re-running the notebook continues with the samples still pending.
"""),
    code("""
%matplotlib widget
from unetultra.review_ui import step_review

panel, stepper = step_review(review, only="pending")   # only=None to go through every sample again
panel
""", tag="interactive"),
    md("## Save and check\n\nSaves anything not yet written and validates the zips."),
    code("""
print(review.save())
for path in ZIPS:
    print(validate_session(path).format())
"""),
    md("""
## Overview grid (optional)

All samples at a glance, 12 per page, to spot anything you want to revisit. To redo one, run
`stepper = step_review(review, only=None)` above, or set it back to pending here and re-run the
review cell.
"""),
    code("""
from unetultra.review_ui import review_grid

review_grid(review, page=0, per_page=12)
"""),
    md("""
## Check against the fake session's known shapes

Only meaningful for zips made by `tools/make_fake_session.py`: the true masks are regenerated
from the same seed and compared with SAM's. The target is Dice > 0.95.
"""),
    code("""
import tempfile
from unetultra.fake_session import make_fake_session
from unetultra.masks import dice
from unetultra.session import Session

for path in ZIPS:
    with Session(path) as made:
        if made.manifest["app_version"] != "fake":
            continue
        seed = int(made.session_id.removeprefix("fake"))
        size = tuple(made.meta(made.sample_ids[0])["image_size"])
        with tempfile.TemporaryDirectory() as tmp:
            truth_zip = make_fake_session(Path(tmp) / "truth.zip", n_samples=len(made.sample_ids), seed=seed,
                                          split=made.split, classes=made.classes, image_size=size, with_masks=True)
            with Session(truth_zip) as truth:
                scores = [dice(made.instances(s) > 0, truth.instances(s) > 0)
                          for s in made.sample_ids if made.instances(s) is not None and truth.meta(s)["objects"]]
        print(f"{path}: Dice min {min(scores):.3f}, mean {sum(scores) / len(scores):.3f} over {len(scores)} samples")
"""),
])

write("02_train", [
    md("""
# 02 · Calibration and training

1. Fit the per-device **lens-position → distance** calibration on the training captures.
2. Train every model listed in `RUNS`. The default is three:
   - `unet_multi`: the U-Net with the focus input,
   - `image_only`: the same U-Net with no focus input (the ablation),
   - `unet_embed`: the U-Net with the focus input and frozen **DINOv2-small** features.

Only sessions whose manifest says `split: train` are used. The test zip is never opened here.
No model ever receives depth as an input. The first `unet_embed` run downloads the DINOv2
weights (about 90 MB).
"""),
    code(SETUP),
    code("""
# checkpoint name -> architecture (any name registered in unetultra.models.ARCHITECTURES)
RUNS = {
    "model": "unet_multi",
    "model_image_only": "image_only",
    "model_embed": "unet_embed",
}
EPOCHS = 60
BATCH_SIZE = 8
DEPTH_WEIGHT = 0.5          # weight of the depth loss against the mask loss
OBJECT_DEPTH_WEIGHT = 1.0   # weight of the depth error averaged over object pixels alone (0 = plain L1 over all valid pixels)
VAL_SESSIONS = None         # e.g. ["../data/session_....zip"]; None holds out about a quarter of the sessions
""", tag="parameters"),
    code("""
from unetultra.session import find_sessions
from unetultra.train import split_sessions

all_train_paths = find_sessions(DATA, "train")
SINGLE_SESSION = len(all_train_paths) == 1 and not VAL_SESSIONS
if SINGLE_SESSION:
    train_paths = val_paths = all_train_paths
    print("One training session:", all_train_paths[0])
    print("Its last 20% of samples are held out for validation. They come from the same room and day,")
    print("so validation numbers will flatter the model; trust only the test session in 03_evaluate.")
else:
    train_paths, val_paths = split_sessions(all_train_paths, VAL_SESSIONS)
    print("train:", *train_paths, sep="\\n  ")
    print("validation:", *val_paths, sep="\\n  ")
"""),
    md("""
## Lens calibration

Each point is one capture: the lens reading after focus settled, against the LiDAR depth at the
tap (median of valid pixels in a small window). The fit is monotonic (isotonic regression on
1/distance). Residuals should scatter around zero with no trend.
"""),
    code("""
from unetultra import plots
from unetultra.calibration import fit_from_sessions

calibration, points = fit_from_sessions(train_paths)
calibration.save(MODELS / "calibration.json")
print(f"{calibration.n_samples} samples, RMSE {calibration.rmse_m * 100:.2f} cm, device {calibration.device_model}")
plots.calibration_fit(calibration, points);
"""),
    md("""
## Datasets

Accepted samples only. Pre-processing takes the centre square of each frame and resizes it to
512×512, with the intrinsics transformed to match. Samples whose object was tapped outside that
square cannot be used and are listed as `excluded_outside_crop`.
"""),
    code("""
from torch.utils.data import Subset
from unetultra.dataset import SessionDataset
from unetultra.models import uses_meta
from unetultra.train import split_single_session


def datasets(architecture):
    \"\"\"Train and validation sets for one architecture (an image-only model gets a zeroed meta input).\"\"\"
    use_meta = uses_meta(architecture)
    train = SessionDataset(train_paths, calibration, augment=True, use_meta=use_meta)
    val = SessionDataset(val_paths, calibration, use_meta=use_meta)
    if SINGLE_SESSION:
        train_indices, val_indices = split_single_session(train)
        return Subset(train, train_indices), Subset(val, val_indices), train
    return train, val, train


train_set, val_set, full_train = datasets("unet_multi")
background = sum(int(train_set[i]["instances"].max()) == 0 for i in range(len(train_set)))
print(f"train {len(train_set)} samples ({full_train.skipped} skipped as not accepted), validation {len(val_set)}")
for path, sample_id, reason in full_train.excluded:
    print(f"  {reason}: {Path(path).name} sample {sample_id}")
print(f"background-only share of training samples: {background / max(len(train_set), 1):.0%} (aim for about 10%)")
print("classes:", full_train.classes)
"""),
    md("## Train\n\nOne line per epoch; `*` marks a new best checkpoint (lowest validation object-distance error)."),
    code("""
from unetultra.train import TrainConfig, fit

histories = {}
for name, architecture in RUNS.items():
    config = TrainConfig(architecture=architecture, epochs=EPOCHS, batch_size=BATCH_SIZE,
                         depth_weight=DEPTH_WEIGHT, object_depth_weight=OBJECT_DEPTH_WEIGHT)
    run_train, run_val, _ = datasets(architecture)
    histories[name], checkpoint = fit(run_train, run_val, config, MODELS, name=name)
    print("best checkpoint:", checkpoint, "\\n")
"""),
    md("## Learning curves"),
    code("""
plots.learning_curves(histories);
"""),
])

write("03_evaluate", [
    md("""
# 03 · Evaluation

Scores every trained model on a **separate test zip** (different room or day). The models
receive the image and the meta vector only; LiDAR depth is used solely as the truth to score
against.

Compared here:
- every checkpoint in `RUNS` (by default the U-Net with focus, the image-only ablation and the
  DINOv2 model),
- a no-learning baseline that takes the calibrated focus distance as the object's distance.
"""),
    code(SETUP),
    code("""
# checkpoint name -> label in the tables; the first one is the reference model
RUNS = {"model": "U-Net + focus", "model_image_only": "image only", "model_embed": "U-Net + DINOv2"}
""", tag="parameters"),
    code("""
from unetultra.calibration import LensCalibration
from unetultra.dataset import SessionDataset
from unetultra.models import uses_meta
from unetultra.session import find_sessions
from unetultra.train import load_model

TEST_ZIPS = find_sessions(DATA, "test")        # only sessions marked split: test
assert TEST_ZIPS, "no test session found in ../data"
calibration = LensCalibration.load(MODELS / "calibration.json")
RUNS = {name: label for name, label in RUNS.items() if (MODELS / f"{name}.pt").exists()}
models, test_sets = {}, {}
for name, label in RUNS.items():
    models[label], checkpoint = load_model(MODELS / f"{name}.pt")
    test_sets[label] = SessionDataset(TEST_ZIPS, calibration, use_meta=uses_meta(checkpoint["architecture"]))
    print(f"{label}: {checkpoint['architecture']}, checkpoint from epoch {checkpoint['epoch']}")
reference = next(iter(RUNS.values()))
test_set = test_sets[reference]
print(f"{len(test_set)} test samples from {len(TEST_ZIPS)} session(s)")
"""),
    md("## Metrics"),
    code("""
import json
from unetultra import metrics
from unetultra.export import headline_metrics

results, summaries, predictions = {}, {}, {}
for (name, label) in RUNS.items():
    rows, image_rows, predictions[label] = metrics.evaluate_model(models[label], test_sets[label])
    results[label], summaries[label] = rows, metrics.summarise(rows, image_rows)
    # Kept next to the checkpoint so the export can copy them into the model card.
    (MODELS / f"{name}_test_metrics.json").write_text(json.dumps(headline_metrics(summaries[label])))
rows, summary = results[reference], summaries[reference]
print(metrics.format_summary(summaries))
"""),
    md("""
### Focus-only baseline

Using the calibrated focus distance as the distance of the object the lens focused on, with no
model at all. If a model cannot beat this, it has learned nothing about distance beyond focus.
"""),
    code("""
baseline = summary["distance_focus_baseline"]
print(f"focused objects: {baseline['n']}")
print(f"  {'focus distance alone':22}: {baseline['mae']:.2f} cm mean abs error ({baseline['mape']:.1f}%)")
for label, s in summaries.items():
    on_targets = s["distance_model_on_focus_targets"]
    print(f"  {label:22}: {on_targets['mae']:.2f} cm mean abs error ({on_targets['mape']:.1f}%)")
"""),
    md("""
### Size: before and after the bias fix

The reference method (2nd–98th percentile between pixel centres) under-reports size by about 4%
and is thrown off by depth outliers at the object's edge. The fix measures outlier pixels at
the object's median depth and uses the full pixel footprint.
"""),
    code("""
print(metrics.format_size_fix(summary))
lidar_vs_ruler = summary["lidar_width_vs_measured"]
if lidar_vs_ruler["n"]:
    print(f"\\nFor reference, LiDAR-derived width against the ruler: {lidar_vs_ruler['mae']:.2f} cm "
          f"({lidar_vs_ruler['mape']:.1f}%) over {lidar_vs_ruler['n']} objects")
"""),
    md("### By distance"),
    code("""
for label, r in results.items():
    print(label)
    for row in metrics.distance_error_by_bin(r):
        print(f"  {row['bin_cm'][0]:>3}-{row['bin_cm'][1]:<3} cm  n={row['n']:<3} "
              f"mean abs {row['mae']:.2f} cm ({row['mape']:.1f}%)  bias {row['bias']:+.2f} cm")
"""),
    md("## Plots"),
    code("""
from unetultra import plots

plots.predicted_vs_true_distance(results)
plots.error_vs_distance(results)
plots.error_vs_distance(results, key="pred_width_cm", truth="true_width_cm", ylabel="width error (cm)")
plots.error_vs_lens(rows);
"""),
    code("""
plots.example_overlays(test_set, predictions[reference], rows);
"""),
    md("## Verdict\n\nGenerated from the numbers above. Add your own reading underneath after looking at the plots."),
    code("""
if "image only" in summaries and reference != "image only":
    print(metrics.verdict(summary, summaries["image only"], rows))
for label, s in summaries.items():
    if label not in (reference, "image only") and s.get("distance", {}).get("n"):
        print(f"{label} against {reference}: distance error {s['distance']['mae']:.2f} cm vs {summary['distance']['mae']:.2f} cm, "
              f"Dice {s['dice']:.3f} vs {summary['dice']:.3f}.")
"""),
    md("_Your notes:_"),
])

write("04_export_coreml", [
    md("""
# 04 · Export model bundles for the phone

Turns each trained checkpoint into a **model bundle** (`exports/<name>.unetmodel`): the Core ML
model, a model card and a self-test, following `docs/model_contract.md`. The app imports
bundles into its model library, and you switch between them on the Inference tab.

This notebook is a thin wrapper around the command-line tool:

    python -m unetultra.export --checkpoint models/model.pt --name unet-multi-512 --out exports/

Run it on the Mac: checking Core ML against PyTorch needs macOS. Run `03_evaluate` first so the
test metrics are copied into each model card.
"""),
    code(SETUP),
    code("""
# checkpoint name -> (bundle name, name shown in the app)
BUNDLES = {
    "model": ("unet-multi-512", "U-Net + focus"),
    "model_image_only": ("unet-image-only-512", "U-Net, image only"),
    "model_embed": ("unet-embed-dinov2s-512", "U-Net + DINOv2-S"),
}
PRECISION = "fp16"   # "fp32" is about twice the size and matches PyTorch more closely
EXPORTS = Path("../exports")
""", tag="parameters"),
    code("""
import json, shutil
from unetultra.calibration import LensCalibration
from unetultra.dataset import SessionDataset
from unetultra.export import export_bundle
from unetultra.models import uses_meta
from unetultra.session import find_sessions
from unetultra.train import load_model

calibration = LensCalibration.load(MODELS / "calibration.json")
zips = find_sessions(DATA, "test") or find_sessions(DATA)
cards = {}
for checkpoint_name, (bundle_name, display_name) in BUNDLES.items():
    checkpoint_path = MODELS / f"{checkpoint_name}.pt"
    if not checkpoint_path.exists():
        print("skipping", checkpoint_name, "(not trained)")
        continue
    architecture = load_model(checkpoint_path)[1]["architecture"]
    samples = SessionDataset(zips, calibration, use_meta=uses_meta(architecture))
    metrics_file = MODELS / f"{checkpoint_name}_test_metrics.json"
    test_metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else None
    bundle, cards[bundle_name], comparison = export_bundle(
        checkpoint_path, bundle_name, EXPORTS, samples, PRECISION, display_name, test_metrics)
    if comparison:
        worst = max(row["depth_max_abs_diff_m"] for row in comparison)
        changed = max(row["mask_pixels_changed"] for row in comparison)
        tolerance = 0.01 if PRECISION == "fp16" else 0.001
        print("  PASS" if worst < tolerance else "  CHECK", f"worst depth difference {worst * 1000:.2f} mm, "
              f"at most {changed} mask pixels changed\\n")
shutil.copy(MODELS / "calibration.json", EXPORTS / "calibration.json")
"""),
    md("## What was exported"),
    code("""
for name, card in cards.items():
    print(f"{card['display_name']:22} {card['architecture']:12} {card['size_mb']:6.1f} MB  {card['precision']}  "
          f"meta {card['meta_features']}  test {card['test_metrics']}")
print("\\nFiles in", EXPORTS.resolve())
for path in sorted(EXPORTS.iterdir()):
    print("  ", path.name)
"""),
    md("""
## Send to the phone

AirDrop the `.unetmodel` files and `calibration.json` from `exports/` to the iPhone (or put them
in iCloud Drive). Open each `.unetmodel` with UNet-ultra, or import it from the app's Models
screen; import `calibration.json` once from the Inference tab. Each import runs the bundle's
self-test on the phone and shows pass or fail. Nothing in `exports/` is committed to git.
"""),
])

write("05_compare_models", [
    md("""
# 05 · Compare exported models

Runs every bundle in `exports/` **through Core ML on the Mac** (the same converted models the
phone runs, not PyTorch) on the same test zip, and puts them side by side: mask quality,
distance and size error, false positives on background-only samples, latency and file size.

Needs macOS. Laptop and phone numbers can still differ a little (different chips, resizing);
the self-test on the phone measures that gap.
"""),
    code(SETUP),
    code("""
EXPORTS = Path("../exports")
""", tag="parameters"),
    code("""
import platform, tempfile
import numpy as np
from unetultra import metrics
from unetultra.calibration import LensCalibration
from unetultra.dataset import SessionDataset
from unetultra.export import coreml_predict, extract_package, read_card
from unetultra.session import find_sessions

assert platform.system() == "Darwin", "Core ML models only run on macOS"
import coremltools as ct

TEST_ZIPS = find_sessions(DATA, "test")
assert TEST_ZIPS, "no test session found in ../data"
calibration = LensCalibration.load(MODELS / "calibration.json")
test_set = SessionDataset(TEST_ZIPS, calibration)

results, summaries, extra = {}, {}, {}
for bundle in sorted(EXPORTS.glob("*.unetmodel")):
    card = read_card(bundle)
    with tempfile.TemporaryDirectory() as tmp:
        mlmodel = ct.models.MLModel(str(extract_package(bundle, tmp)))
        coreml_predict(mlmodel, np.zeros((card["input"]["height"], card["input"]["width"], 3), np.uint8),
                       np.zeros(len(card["meta_features"]), np.float32))   # warm-up, not timed
        rows, image_rows, latencies = metrics.evaluate_predictor(
            lambda image, meta: coreml_predict(mlmodel, image, meta), test_set, card["meta_features"])
    label = card["display_name"]
    results[label], summaries[label] = rows, metrics.summarise(rows, image_rows)
    extra[label] = {"latency_ms": float(np.median(latencies)), "size_mb": card["size_mb"],
                    "architecture": card["architecture"], "precision": card["precision"]}
    print("evaluated", label)
"""),
    md("## Table"),
    code("""
print(metrics.format_summary(summaries))
print()
print(f"{'':38}" + "".join(f"{label:>18}" for label in extra))
print(f"{'Mac latency (median)':38}" + "".join(f"{e['latency_ms']:15.1f} ms" for e in extra.values()))
print(f"{'model size':38}" + "".join(f"{e['size_mb']:15.1f} MB" for e in extra.values()))
print(f"{'precision':38}" + "".join(f"{e['precision']:>18}" for e in extra.values()))
"""),
    md("## Error against distance"),
    code("""
from unetultra import plots

plots.predicted_vs_true_distance(results)
plots.error_vs_distance(results)
plots.error_vs_distance(results, key="pred_width_cm", truth="true_width_cm", ylabel="width error (cm)");
"""),
])
write("06_pets_unet_vs_dino", [
    md("""
# 06 · Plain U-Net against U-Net + DINOv2 on a public dataset

A side experiment, separate from the AirPods project. It asks one question: **do frozen DINOv2
features help the U-Net segment?**

- **Data:** the Oxford-IIIT Pet dataset: about 7,400 photos of cats and dogs, each with a pixel
  mask of the animal. It downloads automatically (about 800 MB) the first time.
- **Models:** the same two architectures as the main project, `unet_multi` (plain U-Net) and
  `unet_embed` (U-Net with a frozen DINOv2-small encoder fused at the bottleneck).
- **What is left out:** this dataset has no depth, focus or focal length. Only the mask is
  trained, and the meta input is zero for both models, so neither gets any camera information.

Both models see the same images, the same split, the same number of epochs and the same seed.
"""),
    code(SETUP),
    code("""
IMAGE_SIZE = 256      # input is IMAGE_SIZE x IMAGE_SIZE; a multiple of 16
TRAIN_LIMIT = 1000    # training photos to use; None uses all (about 3,380 after validation is held out)
VAL_SIZE = 300        # held out of the training split, for choosing the best epoch
TEST_LIMIT = 1000     # test photos to score on; None uses all 3,669
EPOCHS = 15
BATCH_SIZE = 16
PETS_DIR = DATA / "oxford_pets"
""", tag="parameters"),
    md("## Data\n\nThe border between animal and background is marked uncertain in this dataset; those pixels are ignored in training and scoring."),
    code("""
from unetultra import pets

trainval = pets.load_oxford_pets(PETS_DIR, "trainval")
test_base = pets.load_oxford_pets(PETS_DIR, "test")
train_indices, val_indices = pets.split_indices(len(trainval), TRAIN_LIMIT, VAL_SIZE)
train_set = pets.PetSegmentation(trainval, train_indices, IMAGE_SIZE, augment=True)
val_set = pets.PetSegmentation(trainval, val_indices, IMAGE_SIZE)
test_set = pets.PetSegmentation(test_base, range(len(test_base))[: TEST_LIMIT or len(test_base)], IMAGE_SIZE)
print(f"train {len(train_set)}, validation {len(val_set)}, test {len(test_set)} photos at {IMAGE_SIZE}x{IMAGE_SIZE}")
""", tag="data"),
    md("## Train both models\n\nOne line per epoch; `*` marks the best validation Dice so far, which is the checkpoint kept."),
    code("""
models, histories = {}, {}
for architecture, label in pets.ARCHITECTURES.items():
    models[label], histories[label] = pets.fit_segmentation(
        architecture, train_set, val_set, MODELS, epochs=EPOCHS, batch_size=BATCH_SIZE)
    print()
pets.plot_curves(histories);
"""),
    md("## Results on the test photos\n\nNeither model has seen these photos."),
    code("""
import numpy as np

results = {label: pets.evaluate_segmentation(model, test_set, batch_size=BATCH_SIZE) for label, model in models.items()}
print(f"{'':28}" + "".join(f"{label:>18}" for label in results))
print(f"{'mask Dice (mean)':28}" + "".join(f"{r['dice']:18.4f}" for r in results.values()))
print(f"{'mask IoU (mean)':28}" + "".join(f"{r['iou']:18.4f}" for r in results.values()))
print(f"{'photos with Dice < 0.8':28}" + "".join(f"{np.mean(np.array(r['per_image_dice']) < 0.8):18.1%}" for r in results.values()))
print(f"{'time per photo':28}" + "".join(f"{r['ms_per_image']:15.1f} ms" for r in results.values()))
print(f"{'time per training epoch':28}" + "".join(f"{np.mean([row['seconds'] for row in h]):16.0f} s" for h in histories.values()))
counts = {label: pets.count_parameters(model) for label, model in models.items()}
print(f"{'trainable parameters':28}" + "".join(f"{c[0] / 1e6:16.2f} M" for c in counts.values()))
print(f"{'all parameters':28}" + "".join(f"{c[1] / 1e6:16.2f} M" for c in counts.values()))
"""),
    md("""
### Is the difference real?

Both models are scored on the same photos, so the comparison is made photo by photo. The
interval is a 95% bootstrap range for the mean difference: if it does not include zero, the
difference is unlikely to be chance. It says nothing about a different dataset or training budget.
"""),
    code("""
plain, embed = list(results)
mean, low, high = pets.paired_difference(results[plain]["per_image_dice"], results[embed]["per_image_dice"])
print(f"Dice, {embed} minus {plain}: {mean:+.4f}  (95% interval {low:+.4f} to {high:+.4f})")
if low > 0:
    print(f"{embed} is better, beyond what chance would explain on {len(test_set)} photos.")
elif high < 0:
    print(f"{plain} is better, beyond what chance would explain on {len(test_set)} photos.")
else:
    print("The interval includes zero: no clear difference on this run.")
wins = np.mean(np.array(results[embed]["per_image_dice"]) > np.array(results[plain]["per_image_dice"]))
print(f"{embed} has the higher Dice on {wins:.0%} of the photos.")
"""),
    md("## Where they differ most\n\nThe photos with the largest gap in each direction: first where the plain U-Net does better, then where DINOv2 does."),
    code("""
pets.plot_examples(test_set, results, n=6);
"""),
    md("""
## Things to try

- Lower `TRAIN_LIMIT` (for example 100). Pretrained features usually help most when training
  data is scarce, which is the situation in the AirPods project.
- Raise `TRAIN_LIMIT` to `None` and `EPOCHS` to 30 to see whether the plain U-Net catches up.

_Your notes:_
"""),
])
print("notebooks written")
