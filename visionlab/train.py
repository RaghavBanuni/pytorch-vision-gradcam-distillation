"""Training loop with the details that decide whether a result is trustworthy.

Nothing exotic, but nothing skipped either:

* **seeded** - model init, shuffling and augmentation all draw from seeded generators, so a
  reported number can be reproduced.
* **warmup then cosine** - a fresh head with a large learning rate diverges in the first few
  steps; warmup costs nothing and removes that failure mode.
* **label smoothing** - keeps the network from driving logits to infinity, which is also the
  main reason classifiers end up overconfident.  Calibration is measured later, so it matters
  that the training objective is not actively wrecking it.
* **early stopping with restoration** - training stops when validation accuracy stalls, and the
  *best* weights are restored.  Reporting the final epoch's weights after early stopping is a
  quiet way to report a worse model than the one you selected.
* **no augmentation at evaluation** - augmenting a validation set changes the question.

Batching is done over in-memory tensors rather than a DataLoader: the dataset is small, and it
keeps the shuffling deterministic and the code inspectable.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import pandas as pd
import torch
from torch import nn

from .augment import Augmentation
from .data import Batch
from .models import parameter_groups


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 8
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 5e-4
    label_smoothing: float = 0.05
    warmup_epochs: int = 1
    patience: int = 4
    grad_clip: float = 1.0
    head_multiplier: float = 1.0
    seed: int = 0
    device: str = "cpu"
    verbose: bool = False

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0 or self.grad_clip < 0:
            raise ValueError("weight_decay and grad_clip must be non-negative")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must lie in [0, 1)")
        if self.warmup_epochs < 0 or self.warmup_epochs >= max(self.epochs, 1) + 1:
            raise ValueError("warmup_epochs must be non-negative and shorter than training")
        if self.patience < 1:
            raise ValueError("patience must be positive")


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    lr: float
    train_loss: float
    val_loss: float
    val_accuracy: float


@dataclass
class History:
    records: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = 0
    best_val_accuracy: float = 0.0
    stopped_early: bool = False

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([record.__dict__ for record in self.records])

    @property
    def epochs_run(self) -> int:
        return len(self.records)


def set_seed(seed: int) -> torch.Generator:
    """Seed torch and return a generator for shuffling and augmentation."""
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed + 1)
    return generator


def as_tensors(batch: Batch, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.from_numpy(batch.images).to(device=device, dtype=torch.float32)
    labels = torch.from_numpy(batch.labels).to(device=device, dtype=torch.long)
    return images, labels


def batch_indices(
    count: int, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = False
) -> Iterator[torch.Tensor]:
    if count < 1 or batch_size < 1:
        raise ValueError("count and batch_size must be positive")
    order = (
        torch.randperm(count, generator=generator) if shuffle else torch.arange(count)
    )
    for start in range(0, count, batch_size):
        yield order[start : start + batch_size]


@torch.no_grad()
def evaluate_batches(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 128,
    loss_fn: nn.Module | None = None,
) -> tuple[float, float]:
    """Mean loss and accuracy, computed in eval mode without augmentation."""
    criterion = loss_fn or nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    for index in batch_indices(images.shape[0], batch_size):
        chunk, target = images[index], labels[index]
        logits = model(chunk)
        total_loss += float(criterion(logits, target)) * chunk.shape[0]
        correct += int((logits.argmax(dim=1) == target).sum())
    if was_training:
        model.train()
    count = images.shape[0]
    return total_loss / count, correct / count


def _lr_scale(epoch: int, config: TrainConfig) -> float:
    """Linear warmup, then cosine decay to (almost) zero."""
    if config.warmup_epochs > 0 and epoch < config.warmup_epochs:
        return (epoch + 1) / (config.warmup_epochs + 1)
    remaining = max(config.epochs - config.warmup_epochs, 1)
    progress = (epoch - config.warmup_epochs) / remaining
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


class Trainer:
    """Fits a model and returns the history plus the best weights, already restored."""

    def __init__(self, config: TrainConfig | None = None) -> None:
        self.config = config or TrainConfig()
        self.config.validate()

    def fit(
        self,
        model: nn.Module,
        train: Batch,
        val: Batch,
        augment: Augmentation | None = None,
    ) -> History:
        config = self.config
        generator = set_seed(config.seed)
        device = torch.device(config.device)
        model.to(device)

        train_images, train_labels = as_tensors(train, device)
        val_images, val_labels = as_tensors(val, device)

        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        optimizer = torch.optim.AdamW(
            parameter_groups(
                model,
                lr=config.lr,
                head_multiplier=config.head_multiplier,
                weight_decay=config.weight_decay,
            )
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: _lr_scale(epoch, config)
        )

        history = History()
        best_state = copy.deepcopy(model.state_dict())
        best_accuracy = -1.0
        stale = 0

        for epoch in range(config.epochs):
            model.train()
            running = 0.0
            seen = 0
            for index in batch_indices(
                train_images.shape[0], config.batch_size, generator, shuffle=True
            ):
                chunk, target = train_images[index], train_labels[index]
                if augment is not None:
                    chunk = augment(chunk, generator)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(chunk), target)
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                running += float(loss) * chunk.shape[0]
                seen += chunk.shape[0]

            val_loss, val_accuracy = evaluate_batches(model, val_images, val_labels)
            history.records.append(
                EpochRecord(
                    epoch=epoch,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    train_loss=round(running / max(seen, 1), 5),
                    val_loss=round(val_loss, 5),
                    val_accuracy=round(val_accuracy, 5),
                )
            )
            if config.verbose:
                print(
                    f"epoch {epoch:>3}  train {running / max(seen, 1):.4f}  "
                    f"val {val_loss:.4f}  acc {val_accuracy:.4f}"
                )
            scheduler.step()

            if val_accuracy > best_accuracy + 1e-6:
                best_accuracy = val_accuracy
                best_state = copy.deepcopy(model.state_dict())
                history.best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= config.patience:
                    history.stopped_early = True
                    break

        model.load_state_dict(best_state)  # report the model that was selected
        history.best_val_accuracy = round(max(best_accuracy, 0.0), 5)
        return history
