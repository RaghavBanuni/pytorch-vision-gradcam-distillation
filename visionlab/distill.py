"""Knowledge distillation, judged against the control that usually gets skipped.

The student is trained twice: once on hard labels alone, and once with the teacher's softened
logits.  Without that control a distillation result is unreadable - if plain training matches
the distilled student, the teacher contributed nothing and the extra machinery is pure cost.

The loss follows Hinton, Vinyals & Dean: KL divergence between the temperature-softened
distributions, multiplied by ``T^2`` so the gradient magnitude does not shrink as the
temperature rises, blended with the ordinary cross-entropy on the true labels.  Dropping the
``T^2`` term is the classic implementation bug: the loss still trains, just far too slowly to
notice as a bug rather than as "distillation does not help".
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .augment import Augmentation
from .data import Batch
from .models import count_parameters, parameter_groups
from .train import (
    EpochRecord,
    History,
    TrainConfig,
    as_tensors,
    batch_indices,
    evaluate_batches,
    set_seed,
    _lr_scale,
)


@dataclass(frozen=True)
class DistillConfig:
    """Distillation-specific knobs.

    ``alpha`` weights the teacher against the labels.  At 1.0 the student never sees a true
    label, which is fragile if the teacher is wrong; the default keeps a real supervised signal.
    """

    temperature: float = 4.0
    alpha: float = 0.7

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")


def kd_components(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 4.0,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two halves of the objective: ``(soft_loss, hard_loss)``.

    ``soft_loss`` is ``T^2 * KL(teacher_T || student_T)``; it is exactly zero when the student
    reproduces the teacher's distribution, which the tests check.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student_log = F.log_softmax(student_logits / temperature, dim=1)
    teacher_log = F.log_softmax(teacher_logits.detach() / temperature, dim=1)
    soft = F.kl_div(student_log, teacher_log, reduction="batchmean", log_target=True) * (
        temperature**2
    )
    hard = F.cross_entropy(student_logits, targets, label_smoothing=label_smoothing)
    return soft, hard


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Blended distillation loss.  ``alpha=0`` reduces exactly to cross-entropy."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    soft, hard = kd_components(
        student_logits, teacher_logits, targets, temperature, label_smoothing
    )
    return alpha * soft + (1.0 - alpha) * hard


class DistillTrainer:
    """Trains a student against a frozen teacher, otherwise identical to :class:`Trainer`."""

    def __init__(
        self, config: TrainConfig | None = None, distill: DistillConfig | None = None
    ) -> None:
        self.config = config or TrainConfig()
        self.config.validate()
        self.distill = distill or DistillConfig()
        self.distill.validate()

    def fit(
        self,
        student: nn.Module,
        teacher: nn.Module,
        train: Batch,
        val: Batch,
        augment: Augmentation | None = None,
    ) -> History:
        config = self.config
        generator = set_seed(config.seed)
        device = torch.device(config.device)
        student.to(device)
        teacher.to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)  # the teacher is a fixed target, never trained

        train_images, train_labels = as_tensors(train, device)
        val_images, val_labels = as_tensors(val, device)

        optimizer = torch.optim.AdamW(
            parameter_groups(
                student,
                lr=config.lr,
                head_multiplier=config.head_multiplier,
                weight_decay=config.weight_decay,
            )
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: _lr_scale(epoch, config)
        )

        history = History()
        best_state = copy.deepcopy(student.state_dict())
        best_accuracy = -1.0
        stale = 0

        for epoch in range(config.epochs):
            student.train()
            running = 0.0
            seen = 0
            for index in batch_indices(
                train_images.shape[0], config.batch_size, generator, shuffle=True
            ):
                chunk, target = train_images[index], train_labels[index]
                if augment is not None:
                    chunk = augment(chunk, generator)
                with torch.no_grad():
                    teacher_logits = teacher(chunk)
                optimizer.zero_grad(set_to_none=True)
                loss = kd_loss(
                    student(chunk),
                    teacher_logits,
                    target,
                    temperature=self.distill.temperature,
                    alpha=self.distill.alpha,
                    label_smoothing=config.label_smoothing,
                )
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip)
                optimizer.step()
                running += float(loss) * chunk.shape[0]
                seen += chunk.shape[0]

            val_loss, val_accuracy = evaluate_batches(student, val_images, val_labels)
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
                    f"epoch {epoch:>3}  kd {running / max(seen, 1):.4f}  "
                    f"val {val_loss:.4f}  acc {val_accuracy:.4f}"
                )
            scheduler.step()

            if val_accuracy > best_accuracy + 1e-6:
                best_accuracy = val_accuracy
                best_state = copy.deepcopy(student.state_dict())
                history.best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= config.patience:
                    history.stopped_early = True
                    break

        student.load_state_dict(best_state)
        history.best_val_accuracy = round(max(best_accuracy, 0.0), 5)
        return history


def compression_report(teacher: nn.Module, student: nn.Module) -> dict[str, float]:
    """Parameter counts and the compression ratio, so the trade is stated in both directions."""
    teacher_parameters = count_parameters(teacher)
    student_parameters = count_parameters(student)
    if student_parameters == 0:
        raise ValueError("the student has no parameters")
    return {
        "teacher_parameters": teacher_parameters,
        "student_parameters": student_parameters,
        "compression_ratio": round(teacher_parameters / student_parameters, 2),
        "student_share": round(student_parameters / teacher_parameters, 4),
    }
