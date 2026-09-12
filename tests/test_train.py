"""Tests for the training loop.

The test that earns its keep is ``test_the_best_weights_are_restored``: early stopping that
leaves the final epoch's weights in place reports a model nobody selected, and the mistake is
invisible in the logs.
"""

from __future__ import annotations

import pytest
import torch

from visionlab.augment import NO_AUGMENTATION, STANDARD
from visionlab.models import SmallResNet, TinyCNN
from visionlab.train import (
    TrainConfig,
    Trainer,
    _lr_scale,
    as_tensors,
    batch_indices,
    evaluate_batches,
    set_seed,
)


def small_config(**overrides) -> TrainConfig:
    base = {
        "epochs": 3,
        "batch_size": 32,
        "lr": 5e-3,
        "seed": 0,
        "patience": 3,
        "warmup_epochs": 1,
    }
    return TrainConfig(**{**base, **overrides})


def test_batch_indices_cover_every_sample_once():
    chunks = list(batch_indices(10, 4))
    assert [len(chunk) for chunk in chunks] == [4, 4, 2]
    assert sorted(int(value) for chunk in chunks for value in chunk) == list(range(10))


def test_shuffling_is_seeded():
    first = torch.cat(list(batch_indices(10, 4, set_seed(3), shuffle=True)))
    second = torch.cat(list(batch_indices(10, 4, set_seed(3), shuffle=True)))
    assert torch.equal(first, second)
    assert sorted(first.tolist()) == list(range(10))


def test_as_tensors_produces_the_expected_dtypes(datasets):
    images, labels = as_tensors(datasets["val"])
    assert images.dtype == torch.float32 and labels.dtype == torch.long
    assert images.shape[0] == labels.shape[0] == len(datasets["val"])


def test_learning_rate_warms_up_then_decays():
    config = small_config(epochs=10, warmup_epochs=2)
    warmup = [_lr_scale(epoch, config) for epoch in range(2)]
    assert warmup[0] < warmup[1] <= 1.0
    assert _lr_scale(2, config) == pytest.approx(1.0)
    assert _lr_scale(9, config) < 0.1


def test_training_reduces_the_loss(datasets):
    model = SmallResNet(n_classes=4, width=8, blocks=(2,))
    history = Trainer(small_config(epochs=3)).fit(
        model, datasets["train"], datasets["val"], NO_AUGMENTATION
    )
    frame = history.frame()
    assert len(frame) == 3
    assert frame["train_loss"].iloc[-1] < frame["train_loss"].iloc[0]


def test_the_model_learns_something(trained):
    _model, history = trained
    assert history.best_val_accuracy > 0.30  # chance is 0.25


def test_the_best_weights_are_restored(trained, datasets):
    """The returned model must be the one the history says was selected."""
    model, history = trained
    images, labels = as_tensors(datasets["val"])
    _loss, accuracy = evaluate_batches(model, images, labels)
    assert accuracy == pytest.approx(history.best_val_accuracy, abs=1e-4)


def test_history_records_every_epoch(trained):
    _model, history = trained
    assert history.epochs_run == len(history.records)
    assert 0 <= history.best_epoch < history.epochs_run
    frame = history.frame()
    assert list(frame.columns) == ["epoch", "lr", "train_loss", "val_loss", "val_accuracy"]
    assert frame["val_accuracy"].between(0.0, 1.0).all()


def test_training_is_reproducible(datasets):
    results = []
    for _run in range(2):
        model = TinyCNN(n_classes=4, width=4)
        history = Trainer(small_config(epochs=2)).fit(
            model, datasets["train"], datasets["val"], STANDARD
        )
        results.append((history.best_val_accuracy, [record.train_loss for record in history.records]))
    assert results[0] == results[1]


def test_evaluation_does_not_leave_the_model_in_eval_mode(datasets):
    model = TinyCNN(n_classes=4, width=4)
    model.train()
    images, labels = as_tensors(datasets["val"])
    evaluate_batches(model, images, labels)
    assert model.training


def test_configuration_is_validated():
    with pytest.raises(ValueError, match="epochs and batch_size"):
        TrainConfig(epochs=0).validate()
    with pytest.raises(ValueError, match="lr must be positive"):
        TrainConfig(lr=0.0).validate()
    with pytest.raises(ValueError, match="label_smoothing"):
        TrainConfig(label_smoothing=1.0).validate()
    with pytest.raises(ValueError, match="patience"):
        TrainConfig(patience=0).validate()
