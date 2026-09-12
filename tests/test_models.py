"""Tests for the architectures and the fine-tuning helpers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from visionlab.models import (
    SmallResNet,
    TinyCNN,
    build_model,
    count_parameters,
    freeze_backbone,
    parameter_groups,
    target_layer,
)


def test_forward_shapes():
    inputs = torch.randn(3, 3, 32, 32)
    assert SmallResNet(n_classes=4, width=8, blocks=(2, 2))(inputs).shape == (3, 4)
    assert TinyCNN(n_classes=4, width=8)(inputs).shape == (3, 4)


def test_the_network_handles_odd_input_sizes():
    """Adaptive pooling, not a hard-coded flatten - so 33 pixels does not crash it."""
    assert SmallResNet(width=8, blocks=(2, 2))(torch.randn(2, 3, 33, 33)).shape == (2, 4)


def test_the_student_is_much_smaller_than_the_teacher():
    teacher = SmallResNet(n_classes=4, width=32)
    student = TinyCNN(n_classes=4, width=8)
    assert count_parameters(teacher) > 5 * count_parameters(student)


def test_gradients_reach_every_trainable_parameter():
    model = SmallResNet(n_classes=4, width=8, blocks=(2, 2))
    loss = nn.functional.cross_entropy(
        model(torch.randn(4, 3, 32, 32)), torch.tensor([0, 1, 2, 3])
    )
    loss.backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (parameter.grad is None or not parameter.grad.any())
    ]
    assert missing == []


def test_initialisation_is_seeded():
    torch.manual_seed(11)
    first = SmallResNet(width=8, blocks=(2,))
    torch.manual_seed(11)
    second = SmallResNet(width=8, blocks=(2,))
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.allclose(left, right)


def test_freezing_leaves_only_the_head_trainable():
    model = SmallResNet(n_classes=4, width=8, blocks=(2, 2))
    total = count_parameters(model)
    frozen = freeze_backbone(model)
    trainable = count_parameters(model, trainable_only=True)
    assert frozen + trainable == total
    assert all(
        parameter.requires_grad == name.startswith("head")
        for name, parameter in model.named_parameters()
    )


def test_freezing_everything_is_refused():
    """A typo in the module name would otherwise silently train nothing at all."""
    model = TinyCNN(width=4)
    with pytest.raises(ValueError, match="nothing trainable"):
        freeze_backbone(model, keep_trainable=("classifier",))


def test_parameter_groups_give_the_head_a_larger_step():
    model = SmallResNet(n_classes=4, width=8, blocks=(2,))
    groups = parameter_groups(model, lr=1e-3, head_multiplier=10.0, weight_decay=1e-4)
    learning_rates = sorted({group["lr"] for group in groups})
    assert learning_rates == [1e-3, 1e-2]


def test_parameter_groups_never_decay_norms_or_biases():
    model = SmallResNet(n_classes=4, width=8, blocks=(2,))
    groups = parameter_groups(model, lr=1e-3, weight_decay=1e-4)
    for group in groups:
        if group["weight_decay"] == 0.0:
            assert all(parameter.ndim <= 1 for parameter in group["params"])
        else:
            assert all(parameter.ndim > 1 for parameter in group["params"])


def test_parameter_groups_skip_frozen_parameters():
    model = SmallResNet(n_classes=4, width=8, blocks=(2,))
    freeze_backbone(model)
    groups = parameter_groups(model, lr=1e-3)
    included = sum(len(group["params"]) for group in groups)
    assert included == sum(1 for parameter in model.parameters() if parameter.requires_grad)


def test_parameter_groups_validate_their_inputs():
    model = TinyCNN(width=4)
    with pytest.raises(ValueError, match="must be positive"):
        parameter_groups(model, lr=0.0)


def test_target_layer_is_the_last_convolutional_stage():
    model = SmallResNet(width=8, blocks=(2, 2))
    assert target_layer(model) is model.stages[-1]
    student = TinyCNN(width=8)
    assert target_layer(student) is student.features[-1]


def test_target_layer_reports_when_it_cannot_decide():
    with pytest.raises(ValueError, match="target_layer"):
        target_layer(nn.Linear(4, 2))


def test_build_model_resolves_names():
    assert isinstance(build_model("resnet_small", width=8), SmallResNet)
    assert isinstance(build_model("tiny", width=4), TinyCNN)
    with pytest.raises(ValueError, match="unknown model"):
        build_model("vit")


def test_architectures_validate_their_arguments():
    with pytest.raises(ValueError, match="width >= 4"):
        SmallResNet(width=2)
    with pytest.raises(ValueError, match="dropout"):
        SmallResNet(width=8, dropout=1.0)
    with pytest.raises(ValueError, match="width must be at least 2"):
        TinyCNN(width=1)
