"""Make training masks from tap points with SAM 2, and review them (used by 01_masks.ipynb).

SAM is only ever used here, to create labels. It is never used at inference.
"""

import json
from dataclasses import dataclass, field

import numpy as np

from . import session as fmt
from .session import Session, png_bytes, update_session

DEFAULT_SAM_MODEL = "facebook/sam2.1-hiera-small"


def pick_device():
    """Best available torch device: MPS on Apple Silicon, then CUDA, then CPU."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class Sam2Predictor:
    """Point-prompted SAM 2.1 through Hugging Face transformers.

    Runs on the CPU unless another device is asked for: it takes a second or two per object
    there, while SAM 2 on Apple's MPS backend has been seen to stall. Pass device="mps" to try it.
    The image embedding is not cached; one call encodes the image once and decodes one object.
    """

    def __init__(self, model_id=DEFAULT_SAM_MODEL, device="cpu", model=None, processor=None):
        import torch
        from transformers import Sam2Model, Sam2Processor

        self._torch = torch
        self.device = device or pick_device()
        self.processor = processor or Sam2Processor.from_pretrained(model_id)
        self.model = (model or Sam2Model.from_pretrained(model_id)).eval()
        try:
            self.model.to(self.device)
        except Exception:  # an operator missing on MPS
            self.device = "cpu"
            self.model.to("cpu")

    def _run(self, image, points, labels):
        torch = self._torch
        inputs = self.processor(
            images=image,
            input_points=[[[list(map(float, p)) for p in points]]],
            input_labels=[[[int(label) for label in labels]]],
            return_tensors="pt",
        )
        original_sizes = inputs["original_sizes"]
        inputs = {
            k: (v.to(self.device, torch.float32) if v.is_floating_point() else v.to(self.device))
            for k, v in inputs.items()
            if hasattr(v, "to") and k != "original_sizes"
        }
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=True)
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), original_sizes)[0]
        best = int(outputs.iou_scores[0, 0].argmax())
        return masks[0, best].numpy().astype(bool)

    def predict(self, image, points, labels):
        """image: uint8 [H, W, 3]; points: [[x, y], ...] in image pixels; labels: 1 positive, 0 negative.
        Returns a boolean [H, W] mask for the one object the points describe."""
        try:
            return self._run(image, points, labels)
        except (RuntimeError, NotImplementedError):
            if self.device == "cpu":
                raise
            self.device = "cpu"  # MPS lacks an operator: fall back once and stay on CPU
            self.model.to("cpu")
            return self._run(image, points, labels)


def compose_masks(object_masks, class_ids, shape):
    """Combine per-object boolean masks into labels (class id) and instances (1-based object index).

    Where masks overlap, the smaller object wins, since a small mask inside a large one is far
    more likely to be the real boundary."""
    labels = np.zeros(shape, dtype=np.uint8)
    instances = np.zeros(shape, dtype=np.uint8)
    order = sorted(range(len(object_masks)), key=lambda i: -int(object_masks[i].sum()))
    for i in order:
        labels[object_masks[i]] = class_ids[i]
        instances[object_masks[i]] = i + 1
    return labels, instances


def dice(a, b):
    """Dice overlap of two boolean masks (1.0 when both are empty)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else float(2 * (a & b).sum() / total)


@dataclass
class ReviewItem:
    """One sample under review: its masks, review status and any corrective clicks."""

    path: str
    sample_id: str
    meta: dict
    masks: list = field(default_factory=list)  # one boolean [H, W] mask per object
    status: str = "pending"
    dirty: bool = False

    @property
    def key(self):
        return (self.path, self.sample_id)


class MaskReview:
    """Runs SAM over the tap points of one or more sessions and holds the review state.

    Typical use: run() -> look at the grid -> add_click() / set_status() -> save().
    Background-only samples get an empty mask and are accepted automatically."""

    def __init__(self, paths, predictor):
        self.predictor = predictor
        self.paths = [str(p) for p in paths]
        self.items = []
        for path in self.paths:
            with Session(path) as sess:
                for sample_id in sess.sample_ids:
                    item = ReviewItem(path, sample_id, sess.meta(sample_id), status=sess.mask_status(sample_id))
                    instances = sess.instances(sample_id)
                    if instances is not None:  # reviewed before: start from the saved masks
                        item.masks = [instances == i + 1 for i in range(len(item.meta["objects"]))]
                    self.items.append(item)
        self._by_key = {item.key: item for item in self.items}

    def __getitem__(self, key):
        return self._by_key[tuple(key)]

    def image(self, item):
        with Session(item.path) as sess:
            return sess.image(item.sample_id)

    def _predict_object(self, image, obj):
        points = [obj["tap_point_image"]] + [p["point"] for p in obj.get("extra_points", [])]
        labels = [1] + [1 if p["positive"] else 0 for p in obj.get("extra_points", [])]
        return self.predictor.predict(image, points, labels)

    def run(self, redo=False, progress=None):
        """Predict masks for every sample that has none yet (or all of them with redo=True)."""
        for n, item in enumerate(self.items):
            objects = item.meta["objects"]
            if not objects:
                if item.status == "pending":
                    item.status, item.dirty = "accepted", True
                continue
            if item.masks and not redo:
                continue
            if progress:
                progress(n + 1, len(self.items), item.sample_id)
            image = self.image(item)
            item.masks = [self._predict_object(image, obj) for obj in objects]
            item.status, item.dirty = "pending", True

    def add_click(self, key, object_index, x, y, positive=True):
        """Add a corrective click to one object and re-run SAM for it."""
        item = self[key]
        obj = item.meta["objects"][object_index]
        obj.setdefault("extra_points", []).append({"point": [float(x), float(y)], "positive": bool(positive)})
        item.masks[object_index] = self._predict_object(self.image(item), obj)
        item.dirty = True

    def clear_clicks(self, key, object_index):
        """Remove an object's corrective clicks and go back to the mask from the tap alone."""
        item = self[key]
        obj = item.meta["objects"][object_index]
        obj["extra_points"] = []
        item.masks[object_index] = self._predict_object(self.image(item), obj)
        item.dirty = True

    def set_status(self, key, status):
        if status not in fmt.MASK_STATUSES:
            raise ValueError(f"status must be one of {fmt.MASK_STATUSES}")
        item = self[key]
        item.status, item.dirty = status, True

    def accept_all_pending(self):
        for item in self.items:
            if item.status == "pending" and (item.masks or not item.meta["objects"]):
                item.status, item.dirty = "accepted", True

    def composed(self, item):
        """labels and instances arrays for one item at the image size."""
        width, height = item.meta["image_size"]
        class_ids = [obj["class_id"] for obj in item.meta["objects"]]
        return compose_masks(item.masks, class_ids, (height, width))

    def counts(self):
        return {status: sum(item.status == status for item in self.items) for status in fmt.MASK_STATUSES}

    def save(self):
        """Write labels.png, instances.png, updated meta.json and mask_status into each session."""
        for path in self.paths:
            items = [item for item in self.items if item.path == path and item.dirty]
            if not items:
                continue
            with Session(path) as sess:
                manifest = sess.manifest
            manifest.setdefault("mask_status", {})
            files = {}
            for item in items:
                if item.masks or not item.meta["objects"]:
                    labels, instances = self.composed(item)
                    files[Session.sample_path(item.sample_id, fmt.LABELS_FILE)] = png_bytes(labels)
                    files[Session.sample_path(item.sample_id, fmt.INSTANCES_FILE)] = png_bytes(instances)
                files[Session.sample_path(item.sample_id, fmt.META_FILE)] = json.dumps(item.meta, indent=2).encode()
                manifest["mask_status"][item.sample_id] = item.status
                item.dirty = False
            update_session(path, files, manifest)
        return self.counts()


def overlay(image, masks, taps=(), extra_points=(), alpha=0.5):
    """Image with each object's mask tinted, taps as yellow crosses and extra clicks in green/red."""
    from PIL import Image, ImageDraw

    colours = [(255, 60, 60), (60, 140, 255), (255, 200, 40), (80, 220, 120), (220, 90, 255)]
    out = image.astype(np.float32)
    for i, mask in enumerate(masks):
        out[mask] = (1 - alpha) * out[mask] + alpha * np.array(colours[i % len(colours)])
    picture = Image.fromarray(out.astype(np.uint8))
    draw = ImageDraw.Draw(picture)
    r = max(5, image.shape[1] // 120)
    for x, y in taps:
        draw.line([x - r, y, x + r, y], fill=(255, 240, 0), width=3)
        draw.line([x, y - r, x, y + r], fill=(255, 240, 0), width=3)
    for point in extra_points:
        x, y = point["point"]
        colour = (0, 255, 0) if point["positive"] else (255, 0, 0)
        draw.ellipse([x - r // 2, y - r // 2, x + r // 2, y + r // 2], fill=colour, outline=(255, 255, 255))
    return np.asarray(picture)


def main(argv=None):
    """Run SAM over the tap points of session zips from a terminal and save the masks.

        python -m unetultra.masks data/*.zip

    Masks are saved as "pending"; open 01_masks.ipynb afterwards to review them (it then loads
    the saved masks and does not run SAM again). Each sample is announced before SAM starts on
    it, and if one takes longer than --stall-seconds the Python stack is printed, so a stall
    shows exactly where it is stuck.
    """
    import argparse
    import faulthandler
    import sys
    import time

    parser = argparse.ArgumentParser(description="Make masks with SAM 2 from the saved tap points.")
    parser.add_argument("paths", nargs="+", help="session zip(s)")
    parser.add_argument("--model", default=DEFAULT_SAM_MODEL)
    parser.add_argument("--device", default="cpu", help="cpu (default), mps or cuda")
    parser.add_argument("--redo", action="store_true", help="predict again for samples that already have masks")
    parser.add_argument("--limit", type=int, help="stop after this many samples (for a quick check)")
    parser.add_argument("--stall-seconds", type=int, default=120)
    args = parser.parse_args(argv)

    faulthandler.enable()
    started = time.time()

    def say(message):
        print(f"[{time.time() - started:6.0f}s] {message}", flush=True)

    say("reading sessions")
    review = MaskReview(args.paths, predictor=None)
    todo = [item for item in review.items if item.meta["objects"] and (args.redo or not item.masks)]
    if args.limit:
        todo = todo[: args.limit]
    say(f"{len(review.items)} samples, {len(todo)} need masks")
    if todo:
        say(f"loading {args.model} on {args.device} (the first run downloads about 180 MB)")
        review.predictor = Sam2Predictor(args.model, device=args.device)
        say("SAM 2 loaded")
    for n, item in enumerate(todo, start=1):
        say(f"{n}/{len(todo)}  sample {item.sample_id}  ({len(item.meta['objects'])} object(s))")
        faulthandler.dump_traceback_later(args.stall_seconds, file=sys.stderr)
        image = review.image(item)
        item.masks = [review._predict_object(image, obj) for obj in item.meta["objects"]]
        faulthandler.cancel_dump_traceback_later()
        item.status, item.dirty = "pending", True
        if n % 25 == 0:
            review.save()
            say("saved so far")
    for item in review.items:  # background-only samples need no mask
        if not item.meta["objects"] and item.status == "pending":
            item.status, item.dirty = "accepted", True
    say(f"saving: {review.save()}")
    say("done. Review the masks in notebooks/01_masks.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
