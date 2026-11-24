# UNet-ultra

A proof of concept: a segmentation model trained with LiDAR depth as the **label** can, at
inference time and **without LiDAR**, report for a tapped object

1. which pixels are the object,
2. how far away it is (cm),
3. how big it is as seen from the camera (width × height, cm),

from an RGB frame plus two camera readings: the lens focus position and the focal length.

LiDAR is used only to create training labels and as ground truth for scoring. The model never
receives depth as an input. The PoC object is one closed AirPods case in one cover, at about
15–70 cm.

**Status:** all seven phases of the main brief are written. The Python side is tested on
synthetic data; the iOS app has not been compiled yet, and nothing has run on real captures.
See [What has and has not been verified](#what-has-and-has-not-been-verified).

## How it fits together

```
iPhone (Capture tab)  ──zip──▶  Mac notebooks                       ──▶  iPhone (Inference tab)
RGB + LiDAR depth               01_masks      SAM 2 masks from taps       UNetUltra.mlpackage
intrinsics, lens position       02_train      lens calibration + U-Net    calibration.json
tap points                      03_evaluate   separate test zip           mask, distance, size
                                04_export     Core ML                     LiDAR shown as reference
```

- **Masks:** you tap the object on the phone; SAM 2 turns the tap into a mask on the Mac. SAM is
  never used at inference — the trained U-Net produces the mask.
- **Focus:** `lensPosition` (0–1) is not a distance. A per-device monotonic calibration maps it
  to metres; the model sees the calibrated distance, never the raw reading.
- **Geometry is computed, not learned:** the model predicts a mask and per-pixel depth; distance
  and size come from pinhole back-projection with the frame's own fx, fy, cx, cy.
- **The model sees a square.** Frames are captured and stored whole (4:3, for example
  1920×1440). Pre-processing, identical in the Python dataset and in the app, takes the centre
  square of the frame and resizes it to **512×512**; intrinsics and tap points go through the
  same transform (`geometry.CenterSquareCrop`). Depth, labels and instance maps are each
  centre-cropped to their own square and resized with nearest neighbour. An object tapped
  outside the square cannot be used: the app refuses such taps, and already-captured samples
  are reported as `excluded_outside_crop` by the validator and the dataset loader.
- **Splits are by capture session**, never by frame. The test zip comes from a different room or
  day and is never touched during training or calibration.

## Repository layout

```
ios/UNetUltra/          Xcode project (SwiftUI): Capture and Inference tabs
python/unetultra/       package: session, validate, geometry, model, fake_session,
                        dataset, calibration, metrics (the last three arrive in phases 3–4)
notebooks/              01_masks, 02_train, 03_evaluate, 04_export_coreml
tools/make_fake_session.py
tests/                  pytest
data/  models/          gitignored: zips, checkpoints and .mlpackage never get committed
```

## Python setup (Mac, Apple Silicon, Python 3.11)

```bash
python3.11 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -e ".[all]"
```

```bash
pytest
```

`[all]` adds Jupyter, scikit-learn, `transformers` (SAM 2.1) and `coremltools` to the core
dependencies (PyTorch, torchvision, NumPy, SciPy, Pillow).

## Workflow

### 1. Capture on the phone, export, check

Capture a session in the app's **Capture** tab (see [Using the app](#using-the-app)), export the
zip, and put it in `data/`. Then:

```bash
python -m unetultra.validate data/session_<id>.zip
```

```bash
python -m unetultra.capture_report data/session_<id>.zip --out data/report
```

The first command checks every sample against the data contract. The second writes one overlay
per sample (depth edges in red on the photo: they should trace the object's outline) and prints
how many valid LiDAR pixels land on the object at each distance, with a safe minimum distance.

The smallest useful set is **one training session of about 100 samples** and **one test session
of about 20**, captured in a different room or on a different day and marked *test* when you
start it in the app. With a single training session, `02_train` holds out the last 20% of its
samples for validation; those numbers flatter the model, so trust only the test session. Two or
more training sessions give a proper whole-session validation split.

### 2. Notebooks

```bash
jupyter lab notebooks
```

| Notebook | What it does | Output |
| --- | --- | --- |
| `01_masks` | SAM 2.1 mask per tapped object; review grid; extra clicks to fix a mask | `labels.png`, `instances.png`, `mask_status` written into each zip |
| `02_train` | Lens calibration; trains the model and the image-only ablation | `models/calibration.json`, `models/model.pt`, `models/model_image_only.pt` |
| `03_evaluate` | Scores both models and the focus-only baseline on the test zip | metrics, plots, a generated verdict |
| `04_export_coreml` | Converts to Core ML, compares with PyTorch, copies files for the app | `models/UNetUltra.mlpackage` |

The notebooks are generated from `tools/build_notebooks.py`; edit that file and re-run it rather
than editing the `.ipynb` JSON by hand if you want clean diffs.

### 3. Try it without a phone

```bash
python tools/make_fake_session.py data/session_fake_train0.zip --seed 0 --with-masks
```

```bash
python tools/make_fake_session.py data/session_fake_train1.zip --seed 1 --with-masks
```

```bash
python tools/make_fake_session.py data/session_fake_test.zip --seed 9 --split test --with-masks
```

The fake sessions use exactly the real format (flat shapes of known size at 15–70 cm, some
photos with two objects, some background-only), so notebooks 02–04 run on them as they are.
Leave out `--with-masks` to get "pending" masks, as when a zip leaves the phone, and exercise
`01_masks`.

## Xcode setup

Requires the latest stable Xcode and, for real captures, an iPhone 12 Pro or later (LiDAR) on
iOS 17+.

1. Open `ios/UNetUltra/UNetUltra.xcodeproj`.
2. Select the **UNetUltra** target → **Signing & Capabilities** → tick *Automatically manage
   signing* and choose your personal team. Do the same for **UNetUltraTests**.
3. Change the bundle identifier (`com.unetultra.app`) to something unique to you, for example
   `com.<yourname>.UNetUltra`; a personal team cannot sign an identifier someone else has used.
4. Pick a simulator and press ⌘B to build, ⌘U to run the unit tests. On the simulator the app
   reports that LiDAR is unavailable; that is expected.
5. To run on the phone: connect it, select it as the destination, press ⌘R, then on the phone
   trust the developer certificate under Settings → General → VPN & Device Management.
   Personal-team builds expire after 7 days; rebuild to renew.

Command-line build for the simulator:

```bash
xcodebuild -project ios/UNetUltra/UNetUltra.xcodeproj -scheme UNetUltra -destination 'generic/platform=iOS Simulator' build
```

The project uses file-system-synchronised groups (Xcode 16+): any file added under
`ios/UNetUltra/UNetUltra/` is part of the app without editing the project file.

## Using the app

### Capture tab

1. On first launch allow camera access. A session sheet opens: keep the class list
   (`airpods_case`), choose **train** or **test**, and start the session.
2. Keep the case inside the square guide (the region the model sees; the rest is dimmed) and
   **tap it**. The lens focuses there and, once it settles, the sample is saved automatically;
   you feel a short tap when it is. A tap outside the square, a focus that does not settle, or
   a point with no LiDAR reading saves nothing and says why: just tap again.
3. Follow the **guide** under the preview. It shows progress against the session's target
   (100 samples for train, 20 for test; change it in the session sheet), how many samples each
   distance band has (15–25, 25–35, 35–45, 45–55, 55–70 cm, plus "none" for no-object shots),
   and what to capture next.
4. Press **No object** for a background-only sample (aim for about 1 in 10). For several
   objects in one photo, turn off **Tap to capture**, tap each object, then press **Capture**.
5. The ruler button holds optional measured width, height, facing and distance.
6. The gallery button shows thumbnails with tap markers; touch and hold one to delete it.
7. The share button exports `session_<id>.zip` (AirDrop it to the Mac and move it to `data/`).

The question-mark button shows the capture checklist: case closed in the same cover; 15–70 cm
with varied distance, angle and visible face; lying flat, standing and held in the hand; varied
surfaces, backgrounds and lighting.

### Inference tab

1. Run `04_export_coreml`. It copies `UNetUltra.mlpackage` and `calibration.json` into
   `ios/UNetUltra/UNetUltra/Resources/`; rebuild and run the app. (Both can also be imported
   from Files inside the tab.)
2. Point at the case, keep it inside the square guide, and tap it. When focus settles the model
   runs on that frame.
3. The screen shows the mask overlay, the predicted distance and width × height, and beside them
   the LiDAR distance over the same pixels, labelled as a reference the model does not use.
4. **Log result** appends the prediction, the LiDAR reference and any measured values to a CSV
   you can export.

## Data format

One zip per capture session; this is the contract between the app and the notebooks, checked by
`python -m unetultra.validate`.

```
session_<sessionId>.zip
  manifest.json
  samples/<sampleId>/
    image.jpg        RGB frame, sensor orientation, quality 95
    depth_mm.png     16-bit PNG, millimetres, 0 = no reading, sensor orientation
    confidence.png   8-bit PNG, LiDAR confidence 0/1/2 (omitted if unavailable)
    meta.json        intrinsics, lens position, taps, optional ruler measurements
    labels.png       added by 01_masks: class id per pixel (0 = background, 255 = ignore)
    instances.png    added by 01_masks: object index per pixel (1..N in meta.objects order)
```

Images and depth are stored **as the sensor delivers them** (landscape), with the device
orientation recorded separately, so the intrinsics stay valid for the saved image. Class id 0
is always background; the format supports several classes and several objects per photo, while
the PoC trains one class.

Pixel convention in the Python code: integer coordinates are pixel centres. For a frame of
width W and height H, with `S = min(W, H)`, `x0 = (W - S) / 2`, `y0 = (H - S) / 2` and
`s = 512 / S`, the model's intrinsics are `fx * s`, `fy * s`, `(cx - x0 + 0.5) * s - 0.5` and
`(cy - y0 + 0.5) * s - 0.5`, and points move the same way. A horizontal flip maps `cx` to
`511 - cx`.

## What has and has not been verified

| Part | State |
| --- | --- |
| Python package and tests (`pytest`) | Passing, on synthetic data only |
| Notebooks 02–04 | Executed end to end on synthetic sessions |
| Notebook 01 | Executed with a stand-in for SAM; real SAM 2.1 weights have not been run |
| Core ML conversion | Converts; the Core ML vs PyTorch comparison needs macOS and has not been run |
| iOS app (both tabs) and Swift tests | Written, never compiled or run |
| Anything on real captures | Not done: no real session exists yet |

## Honest limits

- **Small object.** The case covers few pixels, so a one-pixel edge error is several percent of
  its size. The LiDAR depth map is coarse (about 256×192), so few depth readings fall on the
  case beyond about 70 cm. The PoC range is therefore about 15–70 cm, and LiDAR quality at the
  near end is checked in phase 1.
- **Focus.** The focus reading carries the most information up close, which suits this range.
  Lens calibration can drift with temperature; the evaluation checks whether errors correlate
  with the lens reading. The reading may also shift with how the phone is tilted.
- **LiDAR-assisted autofocus.** On Pro iPhones the autofocus system can itself use the LiDAR
  scanner, so the lens position may carry LiDAR-derived information even though the model is
  never given depth. The image-only ablation (phase 4) is the result that is free of this.
- **One object.** The PoC learns one specific object in one cover. Other objects or covers need
  their own data.
- **One phone.** Testing on the same iPhone proves "no LiDAR at inference". Another phone model
  would need its own lens calibration and probably some fine-tuning.
- **What "size" means.** Width × height as seen from the camera. It changes with the angle the
  case is held at, and front-to-back depth cannot be observed from one view. A partly hidden or
  hand-covered case measures smaller than it is.
- **Known measurement bias.** The 2nd–98th percentile trim plus measuring between pixel centres
  under-reports size by about 4%. Phase 4 reports the error before and after fixing it.
