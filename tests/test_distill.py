"""Tests for the distillation objective and trainer.

The loss is checked against the closed form rather than against itself: ``alpha=0`` must reduce
exactly to cross-entropy, a student that already matches the teacher must incur exactly zero
soft loss, and the ``T^2`` factor must be present - dropping it is the classic bug that makes
distillation look useless instead of broken.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch.nn import functional as F

from visionlab.augment import NO_AUGMENTATION
from visionlab.distill import (
    DistillConfig,
    DistillTrainer,
    compression_report,
    kd_components,
    kd_loss,
)
from visionlab.models import SmallResNet, TinyCNN, count_parameters
from visionlab.train import TrainConfig

STUDENT_LOGITS = torch.tensor([[2.0, 0.5, -1.0, 0.0], [0.1, 1.5, 0.2, -0.3]])
TEACHER_LOGITS = torch.tensor([[1.5, 0.7, -0.5, 0.1], [0.0, 2.0, 0.1, -0.2]])
TARGETS = torch.tensor([0, 1])


def test_alpha_zero_is_exactly_cross_entropy():
    loss = kd_loss(STUDENT_LOGITS, TEACHER_LOGITS, TARGETS, alpha=0.0)
    assert float(loss) == pytest.approx(float(F.cross_entropy(STUDENT_LOGITS, TARGETS)), rel=1e-6)


def test_matching_the_teacher_costs_nothing():
    soft, _hard = kd_components(TEACHER_LOGITS.clone(), TEACHER_LOGITS, TARGETS, temperature=3.0)
    assert float(soft) == pytest.approx(0.0, abs=1e-6)


def test_the_soft_loss_includes_the_temperature_squared_factor():
    temperature = 4.0
    soft, _hard = kd_components(
        STUDENT_LOGITS, TEACHER_LOGITS, TARGETS, temperature=temperature
    )
    student_log = F.log_softmax(STUDENT_LOGITS / temperature, dim=1)
    teacher_log = F.log_softmax(TEACHER_LOGITS / temperature, dim=1)
    expected = float(
        F.kl_div(student_log, teacher_log, reduction="batchmean", log_target=True)
    ) * temperature**2
    assert float(soft) == pytest.approx(expected, rel=1e-6)
    assert float(soft) > 0.0


def test_a_higher_temperature_softens_the_target():
    """Raising T flattens both distributions, so the KL term shrinks before the T^2 rescale."""
    cold_log = F.log_softmax(STUDENT_LOGITS / 1.0, dim=1)
    cold_teacher = F.log_softmax(TEACHER_LOGITS / 1.0, dim=1)
    warm_log = F.log_softmax(STUDENT_LOGITS / 8.0, dim=1)
    warm_teacher = F.log_softmax(TEACHER_LOGITS / 8.0, dim=1)
    cold = float(F.kl_div(cold_log, cold_teacher, reduction="batchmean", log_target=True))
    warm = float(F.kl_div(warm_log, warm_teacher, reduction="batchmean", log_target=True))
    assert warm < cold


def test_the_blend_sits_between_its_two_terms():
    soft, hard = kd_components(STUDENT_LOGITS, TEACHER_LOGITS, TARGETS, temperature=4.0)
    total = kd_loss(STUDENT_LOGITS, TEACHER_LOGITS, TARGETS, temperature=4.0, alpha=0.7)
    assert float(total) == pytest.approx(0.7 * float(soft) + 0.3 * float(hard), rel=1e-6)


def test_gradients_never_reach_the_teacher():
    student = TinyCNN(n_classes=4, width=4)
    teacher = TinyCNN(n_classes=4, width=8)
    images = torch.randn(4, 3, 32, 32)
    loss = kd_loss(student(images), teacher(images), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert all(parameter.grad is not None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_the_loss_validates_its_arguments():
    with pytest.raises(ValueError, match="same shape"):
        kd_components(STUDENT_LOGITS, TEACHER_LOGITS[:, :2], TARGETS)
    with pytest.raises(ValueError, match="temperature must be positive"):
        kd_components(STUDENT_LOGITS, TEACHER_LOGITS, TARGETS, temperature=0.0)
    with pytest.raises(ValueError, match="alpha"):
        kd_loss(STUDENT_LOGITS, TEACHER_LOGITS, TARGETS, alpha=1.5)
    with pytest.raises(ValueError, match="alpha"):
        DistillConfig(alpha=-0.1).validate()


def test_distillation_trains_the_student_and_leaves_the_teacher_alone(trained, datasets):
    teacher, _history = trained
    before = copy.deepcopy(teacher.state_dict())
    student = TinyCNN(n_classes=4, width=4)

    history = DistillTrainer(
        TrainConfig(epochs=2, batch_size=32, lr=5e-3, seed=0, patience=2),
        DistillConfig(temperature=4.0, alpha=0.7),
    ).fit(student, teacher, datasets["train"], datasets["val"], NO_AUGMENTATION)

    assert history.epochs_run == 2
    assert 0.0 <= history.best_val_accuracy <= 1.0
    after = teacher.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before)


def test_the_compression_report_is_arithmetic():
    teacher = SmallResNet(n_classes=4, width=32)
    student = TinyCNN(n_classes=4, width=8)
    report = compression_report(teacher, student)
    assert report["teacher_parameters"] == count_parameters(teacher)
    assert report["student_parameters"] == count_parameters(student)
    assert report["compression_ratio"] == pytest.approx(
        report["teacher_parameters"] / report["student_parameters"], rel=1e-3
    )
    assert report["compression_ratio"] > 5.0


def test_distillation_config_is_validated():
    with pytest.raises(ValueError, match="temperature must be positive"):
        DistillConfig(temperature=0.0).validate()
