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

Turn the tap points saved on the phone into masks with **SAM 2**, then review them.

1. SAM 2.1 runs once per tapped object, with the tap as a positive point.
2. You look through the grid and mark each sample **accepted** or **rejected**.
3. A bad mask can be fixed with extra positive / negative clicks.
4. `labels.png`, `instances.png`, the extra clicks and `mask_status` are written back into each zip.

SAM is used only here, to make training labels. It is never used at inference.
The first run downloads the SAM 2.1 weights from Hugging Face (about 180 MB for *small*).
"""),
    code(SETUP),
    code("""
from unetultra.session import find_sessions
from unetultra.validate import validate_session

ZIPS = find_sessions(DATA)          # or list the zips you want: ["../data/session_....zip"]
for path in ZIPS:
    print(validate_session(path).format())
"""),
    md("## Run SAM 2 from the tap points\n\nSamples that already have masks are left alone; pass `redo=True` to start again."),
    code("""
from unetultra.masks import Sam2Predictor

predictor = Sam2Predictor("facebook/sam2.1-hiera-small")   # or "facebook/sam2.1-hiera-base-plus"
print("SAM 2 running on", predictor.device)
""", tag="predictor"),
    code("""
from unetultra.masks import MaskReview

review = MaskReview(ZIPS, predictor)
review.run(progress=lambda done, total: print(f"\\r{done}/{total}", end=""))
print()
review.counts()
"""),
    md("""
## Review

Each tile shows the mask tint and the tap (yellow cross). Set every sample to **accepted** or
**rejected**; change `page` to move through the session, or pass `only="pending"`.
Background-only samples have an empty mask and are accepted automatically.
"""),
    code("""
from unetultra.review_ui import review_grid

review_grid(review, page=0, per_page=12)
"""),
    md("""
## Fix a mask

Pick a sample and object, choose *add to mask* or *remove from mask*, and click on the image.
SAM re-runs with every click. The clicks are saved in `meta.json` as `extra_points`.
"""),
    code("""
%matplotlib widget
from unetultra.review_ui import fix_mask

item = next(item for item in review.items if item.meta["objects"])   # or review[(ZIPS[0], "0007")]
fix_mask(review, item.key, object_index=0)
""", tag="interactive"),
    md("## Save\n\nNothing is written until this cell runs. Uncomment the first line to accept everything still pending."),
    code("""
# review.accept_all_pending()
print(review.save())
for path in ZIPS:
    print(validate_session(path).format())
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
2. Train `UNetMulti` (mask + depth) with the calibrated focus distance as a side input.
3. Train the **image-only** ablation (meta input zeroed) for comparison in `03_evaluate`.

Only sessions whose manifest says `split: train` are used, and validation is a whole held-out
session. The test zip is never opened here. The model never receives depth as an input.
"""),
    code(SETUP),
    code("""
EPOCHS = 60
BATCH_SIZE = 8
DEPTH_WEIGHT = 0.5          # weight of the depth loss against the mask loss
OBJECT_DEPTH_WEIGHT = 5.0   # extra weight for depth error on object pixels (1.0 = plain L1 over all valid pixels)
VAL_SESSIONS = None         # e.g. ["../data/session_....zip"]; None holds out about a quarter of the sessions
""", tag="parameters"),
    code("""
from unetultra.session import find_sessions
from unetultra.train import split_sessions

train_paths, val_paths = split_sessions(find_sessions(DATA, "train"), VAL_SESSIONS)
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
    md("## Datasets\n\nAccepted samples only, resized to 512×384 with the intrinsics scaled to match."),
    code("""
from unetultra.dataset import SessionDataset

train_set = SessionDataset(train_paths, calibration, augment=True)
val_set = SessionDataset(val_paths, calibration)
background = sum(int(train_set[i]["instances"].max()) == 0 for i in range(len(train_set)))
print(f"train {len(train_set)} samples ({train_set.skipped} skipped as not accepted), validation {len(val_set)}")
print(f"background-only share of training samples: {background / max(len(train_set), 1):.0%} (aim for about 10%)")
print("classes:", train_set.classes)
"""),
    md("## Train the model"),
    code("""
from unetultra.train import TrainConfig, fit

config = TrainConfig(epochs=EPOCHS, batch_size=BATCH_SIZE, depth_weight=DEPTH_WEIGHT,
                     object_depth_weight=OBJECT_DEPTH_WEIGHT)
history, checkpoint = fit(train_set, val_set, config, MODELS, name="model")
print("best checkpoint:", checkpoint)
"""),
    md("## Train the image-only ablation\n\nSame data and settings, with the meta input zeroed."),
    code("""
from dataclasses import replace

ablation_config = replace(config, use_meta=False)
ablation_history, ablation_checkpoint = fit(
    SessionDataset(train_paths, calibration, augment=True, use_meta=False),
    SessionDataset(val_paths, calibration, use_meta=False),
    ablation_config, MODELS, name="model_image_only")
"""),
    md("## Learning curves"),
    code("""
plots.learning_curves({"with focus": history, "image only": ablation_history});
"""),
])

write("03_evaluate", [
    md("""
# 03 · Evaluation

Scores the trained model on a **separate test zip** (different room or day). The model receives
the image and the meta vector only; LiDAR depth is used solely as the truth to score against.

Compared here:
- the model with the focus input,
- the image-only ablation,
- a no-learning baseline that takes the calibrated focus distance as the object's distance.
"""),
    code(SETUP),
    code("""
from unetultra.calibration import LensCalibration
from unetultra.dataset import SessionDataset
from unetultra.session import find_sessions
from unetultra.train import load_model

TEST_ZIPS = find_sessions(DATA, "test")        # only sessions marked split: test
assert TEST_ZIPS, "no test session found in ../data"
calibration = LensCalibration.load(MODELS / "calibration.json")
model, checkpoint = load_model(MODELS / "model.pt")
ablation, _ = load_model(MODELS / "model_image_only.pt")
test_set = SessionDataset(TEST_ZIPS, calibration)
test_set_image_only = SessionDataset(TEST_ZIPS, calibration, use_meta=False)
print(f"{len(test_set)} test samples from {len(TEST_ZIPS)} session(s); checkpoint from epoch {checkpoint['epoch']}")
"""),
    md("## Metrics"),
    code("""
from unetultra import metrics

rows, image_rows, predictions = metrics.evaluate_model(model, test_set)
rows_ablation, image_rows_ablation, _ = metrics.evaluate_model(ablation, test_set_image_only)
summary = metrics.summarise(rows, image_rows)
summary_ablation = metrics.summarise(rows_ablation, image_rows_ablation)
print(metrics.format_summary({"with focus": summary, "image only": summary_ablation}))
"""),
    md("""
### Focus-only baseline

Using the calibrated focus distance as the distance of the object the lens focused on, with no
model at all. If the model cannot beat this, it has learned nothing about distance beyond focus.
"""),
    code("""
baseline, on_targets = summary["distance_focus_baseline"], summary["distance_model_on_focus_targets"]
print(f"focused objects: {baseline['n']}")
print(f"  focus distance alone : {baseline['mae']:.2f} cm mean abs error ({baseline['mape']:.1f}%)")
print(f"  model with focus     : {on_targets['mae']:.2f} cm mean abs error ({on_targets['mape']:.1f}%)")
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
for name, r in (("with focus", rows), ("image only", rows_ablation)):
    print(name)
    for row in metrics.distance_error_by_bin(r):
        print(f"  {row['bin_cm'][0]:>3}-{row['bin_cm'][1]:<3} cm  n={row['n']:<3} "
              f"mean abs {row['mae']:.2f} cm ({row['mape']:.1f}%)  bias {row['bias']:+.2f} cm")
"""),
    md("## Plots"),
    code("""
from unetultra import plots

results = {"with focus": rows, "image only": rows_ablation}
plots.predicted_vs_true_distance(results)
plots.error_vs_distance(results)
plots.error_vs_distance({"with focus": rows}, key="pred_width_cm", truth="true_width_cm", ylabel="width error (cm)")
plots.error_vs_lens(rows);
"""),
    code("""
plots.example_overlays(test_set, predictions, rows);
"""),
    md("## Verdict\n\nGenerated from the numbers above. Add your own reading underneath after looking at the plots."),
    code("""
print(metrics.verdict(summary, summary_ablation, rows))
"""),
    md("_Your notes:_"),
])

write("04_export_coreml", [
    md("""
# 04 · Core ML export

Converts the trained model to `UNetUltra.mlpackage` (iOS 17+) and puts it next to
`calibration.json`, ready for the app's Inference tab.

- input `image`: 512×384 RGB, the 1/255 scaling used in training is built in
- input `meta`: `[1, 2]` float32 = `[log(calibrated focus distance in m), fx at 512 wide / 1000 − 1]`
- outputs `mask_logits` and `depth` (metres)

Run this on the Mac: comparing Core ML with PyTorch needs macOS.
"""),
    code(SETUP),
    code("""
from unetultra.export import export_coreml
from unetultra.train import load_model

model, checkpoint = load_model(MODELS / "model.pt")
package = MODELS / "UNetUltra.mlpackage"
mlmodel = export_coreml(model, checkpoint["classes"], package)
print("wrote", package, "· classes", checkpoint["classes"])
"""),
    md("## Core ML against PyTorch on real samples"),
    code("""
import platform
from unetultra.calibration import LensCalibration
from unetultra.dataset import SessionDataset
from unetultra.export import compare_with_pytorch
from unetultra.session import find_sessions

DEPTH_TOLERANCE_M = 1e-3
LOGIT_TOLERANCE = 1e-2

if platform.system() != "Darwin":
    print("Core ML models only run on macOS: skipping the comparison on", platform.system())
else:
    calibration = LensCalibration.load(MODELS / "calibration.json")
    samples = SessionDataset(find_sessions(DATA), calibration)
    indices = list(range(0, len(samples), max(1, len(samples) // 8)))[:8]
    comparison = compare_with_pytorch(mlmodel, model, samples, indices)
    for row in comparison:
        print(f"{row['sample_id']}: mask logits {row['mask_logits_max_abs_diff']:.2e}, "
              f"depth {row['depth_max_abs_diff_m'] * 1000:.3f} mm, mask pixels changed {row['mask_pixels_changed']}")
    worst_depth = max(row["depth_max_abs_diff_m"] for row in comparison)
    worst_logit = max(row["mask_logits_max_abs_diff"] for row in comparison)
    print(f"\\nmaximum absolute difference: depth {worst_depth * 1000:.3f} mm, mask logits {worst_logit:.2e}")
    assert worst_depth < DEPTH_TOLERANCE_M and worst_logit < LOGIT_TOLERANCE, "Core ML and PyTorch disagree"
    print("PASS")
"""),
    md("""
## Hand over to the app

Both files must exist. Copy them into `ios/UNetUltra/UNetUltra/Resources/` (Xcode picks them up
automatically and compiles the model), or AirDrop them to the phone and import them from the
Inference tab. Neither file is committed to git.
"""),
    code("""
import shutil

calibration_file = MODELS / "calibration.json"
assert package.exists() and calibration_file.exists()
resources = Path("../ios/UNetUltra/UNetUltra/Resources")
resources.mkdir(exist_ok=True)
shutil.copytree(package, resources / package.name, dirs_exist_ok=True)
shutil.copy(calibration_file, resources / calibration_file.name)
print("copied to", resources.resolve())
"""),
])
print("notebooks written")
