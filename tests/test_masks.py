import shutil

import numpy as np
import pytest

from unetultra.fake_session import make_fake_session
from unetultra.masks import MaskReview, compose_masks, dice
from unetultra.session import Session
from unetultra.validate import validate_session


class DiscPredictor:
    """Stands in for SAM: a disc around each positive point, minus a disc around each negative."""

    def __init__(self, radius=12):
        self.radius, self.calls = radius, 0

    def predict(self, image, points, labels):
        self.calls += 1
        yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
        mask = np.zeros(image.shape[:2], dtype=bool)
        for (x, y), label in zip(points, labels):
            disc = (xx - x) ** 2 + (yy - y) ** 2 <= self.radius**2
            mask = (mask | disc) if label else (mask & ~disc)
        return mask


@pytest.fixture
def pending_zip(tmp_path):
    return make_fake_session(tmp_path / "pending.zip", n_samples=6, image_size=(256, 192), seed=5)


def test_compose_gives_overlap_to_the_smaller_object():
    big = np.zeros((10, 10), bool)
    big[:8, :8] = True
    small = np.zeros((10, 10), bool)
    small[2:4, 2:4] = True
    labels, instances = compose_masks([big, small], [1, 2], (10, 10))
    assert instances[3, 3] == 2 and labels[3, 3] == 2 and instances[6, 6] == 1 and labels[9, 9] == 0


def test_review_round_trip(pending_zip):
    review = MaskReview([pending_zip], DiscPredictor())
    review.run()
    statuses = {item.sample_id: item.status for item in review.items}
    background = [item for item in review.items if not item.meta["objects"]]
    assert background and all(item.status == "accepted" for item in background)
    assert all(s == "pending" for sid, s in statuses.items() if sid not in {b.sample_id for b in background})

    target = next(item for item in review.items if item.meta["objects"])
    before = target.masks[0].sum()
    x, y = target.meta["objects"][0]["tap_point_image"]
    review.add_click(target.key, 0, x + 10, y, positive=True)
    assert target.masks[0].sum() > before
    review.add_click(target.key, 0, x, y, positive=False)
    review.set_status(target.key, "rejected")
    review.accept_all_pending()
    counts = review.save()
    assert counts == {"pending": 0, "accepted": 5, "rejected": 1}

    assert validate_session(pending_zip).ok
    with Session(pending_zip) as sess:
        assert sess.mask_status(target.sample_id) == "rejected"
        meta = sess.meta(target.sample_id)
        assert [p["positive"] for p in meta["objects"][0]["extra_points"]] == [True, False]
        instances = sess.instances(target.sample_id)
        assert (instances == 1).sum() == target.masks[0].sum() or len(meta["objects"]) > 1
        empty = sess.labels(background[0].sample_id)
        assert empty is not None and not empty.any()

    # Reopening starts from the saved masks without calling the predictor again.
    predictor = DiscPredictor()
    again = MaskReview([pending_zip], predictor)
    again.run()
    assert predictor.calls == 0 and again.counts()["rejected"] == 1


def test_masks_match_known_shapes_with_a_perfect_predictor(tmp_path):
    """The Dice > 0.95 acceptance check, with an oracle in place of SAM (weights are not in CI)."""
    truth_zip = make_fake_session(tmp_path / "truth.zip", n_samples=6, image_size=(256, 192), seed=5, with_masks=True)
    pending = make_fake_session(tmp_path / "pending.zip", n_samples=6, image_size=(256, 192), seed=5)

    class Oracle:
        def predict(self, image, points, labels):
            x, y = points[0]
            return truth_instances == truth_instances[int(round(y)), int(round(x))]

    review = MaskReview([pending], Oracle())
    with Session(truth_zip) as truth:
        for item in review.items:
            truth_instances = truth.instances(item.sample_id)
            if item.meta["objects"]:
                image = review.image(item)
                item.masks = [review._predict_object(image, obj) for obj in item.meta["objects"]]
                item.dirty = True
        review.accept_all_pending()
        review.run()
        review.save()
        with Session(pending) as made:
            for sample_id in made.sample_ids:
                assert dice(made.instances(sample_id) > 0, truth.instances(sample_id) > 0) > 0.95
                assert (made.labels(sample_id) == truth.labels(sample_id)).all()


def test_sam2_wrapper_plumbing_with_random_weights():
    """Checks the transformers call path and output shape; real weights are needed for real masks."""
    transformers = pytest.importorskip("transformers")
    from unetultra.masks import Sam2Predictor

    try:
        processor = transformers.Sam2Processor(image_processor=transformers.Sam2ImageProcessorFast())
        model = transformers.Sam2Model(transformers.Sam2Config())
    except Exception as exc:  # API differences between transformers versions
        pytest.skip(f"cannot build an untrained SAM 2: {exc}")
    predictor = Sam2Predictor(model=model, processor=processor, device="cpu")
    image = np.random.default_rng(0).integers(0, 255, (192, 256, 3), dtype=np.uint8)
    mask = predictor.predict(image, [[100.0, 80.0], [20.0, 20.0]], [1, 0])
    assert mask.shape == (192, 256) and mask.dtype == bool
