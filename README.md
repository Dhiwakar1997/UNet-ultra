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

**Status:** phase 0 (scaffold). See [Phases](#phases).

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
pip install -e ".[dev,notebooks]"
```

```bash
pytest
```

Make a synthetic session and check it against the data contract:

```bash
python tools/make_fake_session.py data/session_fake_train.zip --with-masks
```

```bash
python -m unetultra.validate data/session_fake_train.zip
```

The fake session uses exactly the real format (flat shapes of known size at 15–70 cm, some
photos with two objects, some background-only), so every notebook can be built and tested
before real captures exist. Without `--with-masks` the masks are "pending", as they are when a
zip leaves the phone.

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

Pixel convention in the Python code: integer coordinates are pixel centres. Resizing scales the
principal point as `(c + 0.5) * s - 0.5`; a horizontal flip maps `cx` to `W - 1 - cx`.

## Phases

| Phase | What | State |
| --- | --- | --- |
| 0 | Scaffold: package, fake session, validator, tests, Xcode project | done |
| 1 | iOS Capture tab and zip export | |
| 2 | `01_masks.ipynb`: SAM 2 masks from taps, review | |
| 3 | `02_train.ipynb`: lens calibration and training | |
| 4 | `03_evaluate.ipynb`: evaluation on a separate test zip | |
| 5 | `04_export_coreml.ipynb`: Core ML export | |
| 6 | iOS Inference tab | |

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
