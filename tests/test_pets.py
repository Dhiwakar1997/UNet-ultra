"""The public-dataset side experiment, run on a tiny stand-in for Oxford-IIIT Pet."""

import numpy as np
import pytest
import torch
from PIL import Image

from unetultra import pets


class FakePets:
    """(image, trimap) pairs like torchvision's OxfordIIITPet: a bright blob with a border ring."""

    def __init__(self, n=24, seed=0):
        rng = np.random.default_rng(seed)
        self.items = []
        for _ in range(n):
            height, width = int(rng.integers(60, 90)), int(rng.integers(60, 90))
            yy, xx = np.mgrid[:height, :width]
            cy, cx, radius = rng.uniform(25, height - 25), rng.uniform(25, width - 25), rng.uniform(12, 20)
            distance = np.hypot(yy - cy, xx - cx)
            image = rng.integers(0, 80, (height, width, 3)).astype(np.uint8)
            image[distance < radius] = (220, 180, 60)
            trimap = np.full((height, width), pets.TRIMAP_BACKGROUND, dtype=np.uint8)
            trimap[distance < radius + 2] = pets.TRIMAP_BORDER
            trimap[distance < radius - 2] = pets.TRIMAP_PET
            self.items.append((Image.fromarray(image), Image.fromarray(trimap)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def test_trimap_conversion_and_dataset_items():
    labels = pets.trimap_to_labels(np.array([[1, 2, 3]]))
    assert labels.tolist() == [[1, 0, 255]]
    dataset = pets.PetSegmentation(FakePets(4), size=64)
    item = dataset[0]
    assert item["image"].shape == (3, 64, 64) and item["labels"].shape == (64, 64)
    assert set(item["labels"].unique().tolist()) == {0, 1, 255} and not item["meta"].any()


def test_split_is_disjoint_and_limited():
    train, val = pets.split_indices(100, train_limit=30, val_size=20)
    assert len(train) == 30 and len(val) == 20 and not set(train) & set(val)
    assert len(pets.split_indices(100, None, 20)[0]) == 80


def test_scores_ignore_the_border():
    labels = torch.tensor([[[1, 1, 255, 0]]])
    perfect_but_for_border = torch.tensor([[[1, 1, 1, 0]]])
    assert pets.dice_iou_per_image(perfect_but_for_border, labels) == ([1.0], [1.0])
    dice, iou = pets.dice_iou_per_image(torch.tensor([[[1, 0, 0, 0]]]), labels)
    assert dice == [pytest.approx(2 / 3)] and iou == [pytest.approx(0.5)]


@pytest.mark.parametrize("architecture", ["unet_multi", "unet_embed"])
def test_both_models_train_and_evaluate(architecture, tmp_path):
    base = FakePets(24)
    train_indices, val_indices = pets.split_indices(len(base), train_limit=12, val_size=6)
    train = pets.PetSegmentation(base, train_indices, size=64, augment=True)
    val = pets.PetSegmentation(base, val_indices, size=64)
    model, history = pets.fit_segmentation(architecture, train, val, tmp_path, epochs=2, batch_size=4,
                                           device="cpu", pretrained=False, log=lambda *_: None)
    assert len(history) == 2 and (tmp_path / f"pets_{architecture}.pt").exists()
    if architecture == "unet_embed":
        assert model.dino.grid == 4  # 64 / 16: the encoder grid matches the bottleneck grid
    scores = pets.evaluate_segmentation(model, val, "cpu", batch_size=4)
    assert 0 <= scores["dice"] <= 1 and len(scores["predictions"]) == len(val) == len(scores["per_image_dice"])
    trainable, total = pets.count_parameters(model)
    assert 0 < trainable <= total


def test_paired_difference_interval():
    a = np.full(50, 0.80)
    mean, low, high = pets.paired_difference(a, a + 0.05)
    assert mean == pytest.approx(0.05) and low == pytest.approx(0.05) and high == pytest.approx(0.05)
    rng = np.random.default_rng(0)
    mean, low, high = pets.paired_difference(a, a + rng.normal(0, 0.05, 50))
    assert low < 0 < high  # pure noise: the interval straddles zero
