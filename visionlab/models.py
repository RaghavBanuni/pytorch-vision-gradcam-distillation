"""Models: a small ResNet, a much smaller student, and an optional torchvision backbone.

The default architecture is self-contained on purpose - nothing here needs a network call, so
the repository runs identically offline.  Transfer learning from real pretrained weights is
supported through :func:`resnet18_backbone`, which imports torchvision lazily so the extra
dependency is only required by the people who ask for it.

Every model exposes ``target_layer``: the module whose output Grad-CAM should read.  Hard-coding
that choice in the explainability code is how CAM implementations end up silently pointing at
the wrong tensor after a refactor.
"""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    """Pre-activation-free residual block, as in the original ResNet."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        out = self.activation(self.norm1(self.conv1(inputs)))
        out = self.norm2(self.conv2(out))
        return self.activation(out + residual)


class SmallResNet(nn.Module):
    """Three-stage residual network sized for small images."""

    def __init__(
        self,
        n_classes: int = 4,
        width: int = 32,
        blocks: tuple[int, ...] = (2, 2, 2),
        in_channels: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if width < 4 or n_classes < 2 or not blocks:
            raise ValueError("width >= 4, n_classes >= 2 and at least one stage are required")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        stages: list[nn.Module] = []
        channels = width
        for stage_index, count in enumerate(blocks):
            out_channels = width * (2**stage_index)
            layers = []
            for block_index in range(count):
                layers.append(
                    BasicBlock(
                        channels if block_index == 0 else out_channels,
                        out_channels,
                        stride=2 if (block_index == 0 and stage_index > 0) else 1,
                    )
                )
            channels = out_channels
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(channels, n_classes),
        )
        self.feature_channels = channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stages(self.stem(inputs))
        pooled = torch.flatten(self.pool(features), 1)
        return self.head(pooled)

    @property
    def target_layer(self) -> nn.Module:
        """Last convolutional stage - the standard Grad-CAM read point."""
        return self.stages[-1]


class TinyCNN(nn.Module):
    """The distillation student: same interface, roughly an order of magnitude smaller."""

    def __init__(self, n_classes: int = 4, width: int = 8, in_channels: int = 3) -> None:
        super().__init__()
        if width < 2:
            raise ValueError("width must be at least 2")

        def block(inputs: int, outputs: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(inputs, outputs, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, width), block(width, width * 2), block(width * 2, width * 4)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(width * 4, n_classes)
        self.feature_channels = width * 4

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = torch.flatten(self.pool(self.features(inputs)), 1)
        return self.head(pooled)

    @property
    def target_layer(self) -> nn.Module:
        return self.features[-1]


def resnet18_backbone(n_classes: int = 4, pretrained: bool = False) -> nn.Module:
    """torchvision ResNet-18 with a fresh head, for genuine transfer learning.

    ``pretrained=True`` downloads ImageNet weights (~45 MB).  Nothing else in this repository
    requires network access, so the download stays opt-in.
    """
    try:
        from torchvision import models  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the resnet18 backbone needs torchvision: pip install 'visionlab[backbones]'"
        ) from error

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model.target_layer = model.layer4  # type: ignore[attr-defined]
    return model


def target_layer(model: nn.Module) -> nn.Module:
    """Resolve the Grad-CAM read point for any supported model."""
    layer = getattr(model, "target_layer", None)
    if isinstance(layer, nn.Module):
        return layer
    for name in ("layer4", "stages", "features"):
        candidate = getattr(model, name, None)
        if isinstance(candidate, nn.Module):
            return candidate
    raise ValueError(
        "cannot determine a Grad-CAM target layer; expose a `target_layer` attribute"
    )


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad or not trainable_only
        )
    )


def freeze_backbone(model: nn.Module, keep_trainable: tuple[str, ...] = ("head", "fc")) -> int:
    """Freeze everything except the named modules; returns the number of frozen parameters.

    Fine-tuning a head on frozen features is the correct first move with a small dataset, and
    the count is returned so the CLI can state exactly how much of the network is actually
    learning rather than implying it.
    """
    frozen = 0
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in keep_trainable):
            parameter.requires_grad_(True)
            continue
        parameter.requires_grad_(False)
        frozen += parameter.numel()
    if frozen == count_parameters(model):
        raise ValueError(
            f"freezing left nothing trainable; none of {keep_trainable} matched a parameter name"
        )
    return int(frozen)


def parameter_groups(
    model: nn.Module,
    lr: float,
    head_multiplier: float = 10.0,
    weight_decay: float = 5e-4,
    head_names: tuple[str, ...] = ("head", "fc"),
) -> list[dict]:
    """Discriminative learning rates, and no weight decay on norms or biases.

    Two conventions that matter in practice: a freshly initialised head needs a larger step
    than pretrained features, and decaying BatchNorm parameters degrades accuracy for no
    benefit.
    """
    if lr <= 0 or head_multiplier <= 0 or weight_decay < 0:
        raise ValueError("lr and head_multiplier must be positive, weight_decay non-negative")

    groups: dict[str, dict] = {
        "backbone_decay": {"params": [], "lr": lr, "weight_decay": weight_decay},
        "backbone_plain": {"params": [], "lr": lr, "weight_decay": 0.0},
        "head_decay": {"params": [], "lr": lr * head_multiplier, "weight_decay": weight_decay},
        "head_plain": {"params": [], "lr": lr * head_multiplier, "weight_decay": 0.0},
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_head = any(name.startswith(prefix) for prefix in head_names)
        no_decay = parameter.ndim <= 1  # biases and normalisation scales
        key = f"{'head' if is_head else 'backbone'}_{'plain' if no_decay else 'decay'}"
        groups[key]["params"].append(parameter)
    return [group for group in groups.values() if group["params"]]


MODEL_FACTORIES = {
    "resnet_small": SmallResNet,
    "tiny": TinyCNN,
}


def build_model(name: str, n_classes: int = 4, pretrained: bool = False, **kwargs) -> nn.Module:
    """Single place where model names are resolved."""
    if name == "resnet18":
        return resnet18_backbone(n_classes=n_classes, pretrained=pretrained)
    try:
        factory = MODEL_FACTORIES[name]
    except KeyError as error:
        raise ValueError(
            f"unknown model {name!r}; choose from {sorted(MODEL_FACTORIES) + ['resnet18']}"
        ) from error
    return factory(n_classes=n_classes, **kwargs)
