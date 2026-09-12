"""Tensor augmentations, including the ones that break the shortcut.

Everything operates on a batched float tensor ``(B, 3, H, W)`` in ``[0, 1]`` and is written
directly in torch, so the package needs no torchvision and the behaviour is easy to test.

The interesting knob is ``cue_break``.  A per-image random *per-channel* colour shift leaves
the shape intact while destroying the constant background tint that identifies a class, so the
shortcut stops paying during training.  It is the cheap intervention: you do not have to fix
the data collection, but you do have to know the shortcut exists - which is what the Grad-CAM
score and the inverted-cue test set are for.

Random draws are taken on the CPU with an explicit ``torch.Generator`` and then moved to the
image's device.  That keeps runs reproducible and avoids the generator/device mismatch that
breaks naive augmentation code the moment it runs on a GPU.

Augmentation applies to training batches only.  Augmenting an evaluation set would change the
question being asked, and quietly improve the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _uniform(shape, generator: torch.Generator, device) -> torch.Tensor:
    """Uniform ``[0, 1)`` draws from a CPU generator, placed on ``device``."""
    return torch.rand(shape, generator=generator).to(device)


def _symmetric(shape, generator: torch.Generator, device) -> torch.Tensor:
    """Uniform draws in ``[-1, 1)``."""
    return _uniform(shape, generator, device) * 2.0 - 1.0


def random_horizontal_flip(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    flip = _uniform(images.shape[0], generator, images.device) < 0.5
    if not bool(flip.any()):
        return images
    flipped = images.clone()
    flipped[flip] = torch.flip(images[flip], dims=[-1])
    return flipped


def random_translate(
    images: torch.Tensor, max_shift: int, generator: torch.Generator
) -> torch.Tensor:
    """Shift each image by up to ``max_shift`` pixels, padding with the edge value.

    Edge padding rather than zeros: a black border would be a new, perfectly predictive
    artefact of augmentation, which is its own kind of shortcut.
    """
    if max_shift < 1:
        return images
    padded = torch.nn.functional.pad(
        images, (max_shift, max_shift, max_shift, max_shift), mode="replicate"
    )
    height, width = images.shape[-2:]
    offsets = torch.randint(0, 2 * max_shift + 1, (images.shape[0], 2), generator=generator)
    out = torch.empty_like(images)
    for index in range(images.shape[0]):
        top, left = int(offsets[index, 0]), int(offsets[index, 1])
        out[index] = padded[index, :, top : top + height, left : left + width]
    return out


def random_erasing(
    images: torch.Tensor,
    probability: float,
    generator: torch.Generator,
    max_fraction: float = 0.25,
) -> torch.Tensor:
    """Occlude a random patch, so the model cannot rely on one part of the shape."""
    if probability <= 0:
        return images
    if not 0.0 < max_fraction < 1.0:
        raise ValueError("max_fraction must lie in (0, 1)")
    out = images.clone()
    height, width = images.shape[-2:]
    patch_h = max(1, int(height * max_fraction))
    patch_w = max(1, int(width * max_fraction))
    draws = torch.rand(images.shape[0], generator=generator)
    for index in range(images.shape[0]):
        if float(draws[index]) >= probability:
            continue
        top = int(torch.randint(0, height - patch_h + 1, (1,), generator=generator).item())
        left = int(torch.randint(0, width - patch_w + 1, (1,), generator=generator).item())
        filler = float(torch.rand(1, generator=generator).item())
        out[index, :, top : top + patch_h, left : left + patch_w] = filler
    return out


def colour_jitter(images: torch.Tensor, strength: float, generator: torch.Generator) -> torch.Tensor:
    """Per-image brightness and contrast jitter, identical across channels.

    This does *not* break the cue: scaling all channels together leaves the relative tint, and
    therefore the shortcut, perfectly readable.  That is why it is separate from ``break_cue``
    - conflating the two is how an ablation ends up proving nothing.
    """
    if strength <= 0:
        return images
    count = images.shape[0]
    brightness = 1.0 + strength * _symmetric((count, 1, 1, 1), generator, images.device)
    shift = 0.5 * strength * _symmetric((count, 1, 1, 1), generator, images.device)
    return (images * brightness + shift).clamp(0.0, 1.0)


def break_cue(images: torch.Tensor, strength: float, generator: torch.Generator) -> torch.Tensor:
    """Randomise the per-channel colour balance, destroying the background tint signal."""
    if strength <= 0:
        return images
    shape = (images.shape[0], images.shape[1], 1, 1)
    offsets = strength * _symmetric(shape, generator, images.device)
    gains = 1.0 + strength * _symmetric(shape, generator, images.device)
    return (images * gains + offsets).clamp(0.0, 1.0)


@dataclass(frozen=True)
class Augmentation:
    """Composed training-time augmentation with an explicit random generator."""

    flip: bool = True
    translate: int = 4
    erase_probability: float = 0.25
    jitter: float = 0.15
    cue_break: float = 0.0

    def validate(self) -> None:
        if self.translate < 0:
            raise ValueError("translate must be non-negative")
        if not 0.0 <= self.erase_probability <= 1.0:
            raise ValueError("erase_probability must lie in [0, 1]")
        if self.jitter < 0 or self.cue_break < 0:
            raise ValueError("jitter and cue_break must be non-negative")
        if self.cue_break > 0.6:
            raise ValueError("a cue_break above 0.6 destroys the shape as well as the tint")

    def __call__(self, images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        self.validate()
        out = images
        if self.flip:
            out = random_horizontal_flip(out, generator)
        out = random_translate(out, self.translate, generator)
        out = colour_jitter(out, self.jitter, generator)
        out = break_cue(out, self.cue_break, generator)
        out = random_erasing(out, self.erase_probability, generator)
        return out.clamp(0.0, 1.0)


NO_AUGMENTATION = Augmentation(
    flip=False, translate=0, erase_probability=0.0, jitter=0.0, cue_break=0.0
)
STANDARD = Augmentation()
CUE_BREAKING = Augmentation(
    flip=True, translate=4, erase_probability=0.25, jitter=0.15, cue_break=0.35
)
