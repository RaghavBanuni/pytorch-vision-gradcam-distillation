"""Tests for Grad-CAM.

Most Grad-CAM code is only ever checked by looking at a heatmap, which cannot fail.  Here the
maths is pinned down with a model whose correct explanation is known in advance: its single
feature channel *is* mean pixel intensity, so the map for the positively-weighted class must
peak on the bright square, and the map for the negatively-weighted class must be empty after
the ReLU.  If the weighting, the sign or the interpolation were wrong, these tests fail.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from visionlab.gradcam import (
    GradCAM,
    box_area_share,
    compute_cams,
    localization_report,
    mask_energy_inside,
    pointing_game,
)


class BrightnessModel(nn.Module):
    """Feature channel = mean intensity. Class 0 weights it +1, class 1 weights it -1."""

    def __init__(self) -> None:
        super().__init__()
        convolution = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            convolution.weight.fill_(1.0 / 3.0)
        self.features = nn.Sequential(convolution, nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(1, 2)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[1.0], [-1.0]]))
            self.head.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(torch.flatten(self.pool(self.features(inputs)), 1))

    @property
    def target_layer(self) -> nn.Module:
        return self.features


BOX = (4, 5, 9, 10)  # x0, y0, x1, y1


def bright_square(size: int = 16) -> tuple[torch.Tensor, np.ndarray]:
    image = torch.full((1, 3, size, size), 0.1)
    x0, y0, x1, y1 = BOX
    image[:, :, y0 : y1 + 1, x0 : x1 + 1] = 0.9
    return image, np.array([BOX], dtype=np.int64)


def test_the_map_peaks_on_the_object():
    image, boxes = bright_square()
    with GradCAM(BrightnessModel()) as explainer:
        cams, logits = explainer(image, targets=torch.tensor([0]))
    assert cams.shape == (1, 16, 16)
    assert logits.shape == (1, 2)
    assert pointing_game(cams, boxes) == 1.0
    assert mask_energy_inside(cams, boxes) > 0.5
    assert mask_energy_inside(cams, boxes) > box_area_share(boxes, 16, 16)


def test_maps_are_scaled_to_one():
    image, _boxes = bright_square()
    with GradCAM(BrightnessModel()) as explainer:
        cams, _logits = explainer(image, targets=torch.tensor([0]))
    assert float(cams.min()) >= 0.0
    assert float(cams.max()) == pytest.approx(1.0)


def test_the_opposing_class_has_no_positive_evidence():
    """Grad-CAM is signed: for class 1 the weight is negative and the ReLU empties the map."""
    image, _boxes = bright_square()
    with GradCAM(BrightnessModel()) as explainer:
        cams, _logits = explainer(image, targets=torch.tensor([1]))
    assert float(cams.max()) == pytest.approx(0.0)


def test_it_works_inside_a_no_grad_block():
    """Evaluation code is full of no_grad; a CAM that silently returns zeros there is a trap."""
    image, boxes = bright_square()
    with torch.no_grad():
        with GradCAM(BrightnessModel()) as explainer:
            cams, _logits = explainer(image, targets=torch.tensor([0]))
    assert float(cams.max()) > 0.0
    assert pointing_game(cams, boxes) == 1.0


def test_it_works_when_every_parameter_is_frozen():
    image, boxes = bright_square()
    model = BrightnessModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with GradCAM(model) as explainer:
        cams, _logits = explainer(image, targets=torch.tensor([0]))
    assert pointing_game(cams, boxes) == 1.0


def test_the_default_target_is_the_predicted_class():
    image, boxes = bright_square()
    model = BrightnessModel()
    with GradCAM(model) as explainer:
        cams, logits = explainer(image)
    assert int(logits.argmax(dim=1)) == 0  # the bright square makes class 0 the prediction
    assert pointing_game(cams, boxes) == 1.0


def test_the_hook_is_removed_on_exit():
    model = BrightnessModel()
    with GradCAM(model) as explainer:
        assert len(explainer.layer._forward_hooks) == 1
    assert len(model.features._forward_hooks) == 0


def test_the_model_is_left_in_training_mode_if_it_started_there():
    image, _boxes = bright_square()
    model = BrightnessModel()
    model.train()
    with GradCAM(model) as explainer:
        explainer(image, targets=torch.tensor([0]))
    assert model.training


def test_bad_input_is_rejected():
    model = BrightnessModel()
    with GradCAM(model) as explainer:
        with pytest.raises(ValueError, match=r"\(B, C, H, W\)"):
            explainer(torch.randn(3, 16, 16))
        with pytest.raises(ValueError, match="one class index per image"):
            explainer(torch.randn(2, 3, 16, 16), targets=torch.tensor([0]))


# ------------------------------------------------------------------- scoring
def test_pointing_game_checks_the_peak_pixel():
    cams = torch.zeros(2, 8, 8)
    cams[0, 3, 5] = 1.0  # row 3, column 5
    cams[1, 0, 0] = 1.0
    boxes = np.array([[4, 2, 6, 4], [4, 2, 6, 4]], dtype=np.int64)
    assert pointing_game(cams[:1], boxes[:1]) == 1.0
    assert pointing_game(cams[1:], boxes[1:]) == 0.0
    assert pointing_game(cams, boxes) == 0.5


def test_mask_energy_is_the_share_inside_the_box():
    cams = torch.ones(1, 4, 4)
    boxes = np.array([[0, 0, 1, 1]], dtype=np.int64)
    assert mask_energy_inside(cams, boxes) == pytest.approx(0.25)


def test_an_empty_map_scores_zero_energy():
    assert mask_energy_inside(torch.zeros(1, 4, 4), np.array([[0, 0, 1, 1]])) == 0.0


def test_box_area_share_is_the_random_baseline():
    boxes = np.array([[0, 0, 1, 1]], dtype=np.int64)
    assert box_area_share(boxes, 4, 4) == pytest.approx(0.25)


def test_scoring_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="align"):
        pointing_game(torch.zeros(2, 4, 4), np.array([[0, 0, 1, 1]]))


def test_localization_report_covers_every_class(trained, datasets):
    model, _history = trained
    summary, per_class = localization_report(model, datasets["test"])
    assert summary["images"] == len(datasets["test"])
    for key in ("pointing_game", "random_baseline", "energy_inside_box", "accuracy"):
        assert 0.0 <= summary[key] <= 1.0
    assert len(per_class) == 4
    assert per_class["images"].sum() == len(datasets["test"])


def test_cams_are_produced_for_every_image(trained, datasets):
    model, _history = trained
    cams, logits = compute_cams(model, datasets["val"], batch_size=32)
    assert cams.shape == (len(datasets["val"]), *datasets["val"].images.shape[-2:])
    assert logits.shape == (len(datasets["val"]), 4)
    assert float(cams.min()) >= 0.0 and float(cams.max()) <= 1.0
