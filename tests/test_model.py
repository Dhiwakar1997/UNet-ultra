import pytest
import torch

from unetultra.model import INPUT_HEIGHT, INPUT_WIDTH, UNetMulti


@pytest.mark.parametrize("n_classes, mask_channels", [(1, 1), (3, 4)])
def test_output_shapes(n_classes, mask_channels):
    model = UNetMulti(n_classes=n_classes).eval()
    with torch.no_grad():
        mask, depth = model(torch.rand(2, 3, INPUT_HEIGHT, INPUT_WIDTH), torch.zeros(2, 2))
    assert mask.shape == (2, mask_channels, INPUT_HEIGHT, INPUT_WIDTH)
    assert depth.shape == (2, 1, INPUT_HEIGHT, INPUT_WIDTH)
    assert (depth > 0).all()


def test_meta_input_changes_output():
    # Train mode, so batch norm keeps the bottleneck signal alive in an untrained network.
    torch.manual_seed(0)
    model = UNetMulti().train()
    image = torch.rand(2, 3, 96, 128)
    with torch.no_grad():
        _, near = model(image, torch.tensor([[-1.5, -0.6]] * 2))
        _, far = model(image, torch.tensor([[0.5, -0.6]] * 2))
    assert (near - far).abs().max() > 1e-4
