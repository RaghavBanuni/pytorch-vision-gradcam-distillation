"""Tests for the augmentations.

The important distinction is asserted directly: ``colour_jitter`` leaves the background tint -
and therefore the shortcut - readable, while ``break_cue`` destroys it.  If those two were
conflated, the intervention comparison in the CLI would be meaningless.
"""

from __future__ import annotations

import torch

from visionlab.augment import (
    CUE_BREAKING,
    NO_AUGMENTATION,
    STANDARD,
    Augmentation,
    break_cue,
    colour_jitter,
    random_erasing,
    random_horizontal_flip,
    random_translate,
)

import pytest


def generator(seed: int = 0) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


@pytest.fixture()
def images(datasets) -> torch.Tensor:
    return torch.from_numpy(datasets["train"].images[:16]).clone()


def test_augmentation_preserves_shape_and_range(images):
    out = STANDARD(images, generator(1))
    assert out.shape == images.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_no_augmentation_is_the_identity(images):
    out = NO_AUGMENTATION(images, generator(1))
    assert torch.allclose(out, images)


def test_augmentation_is_reproducible(images):
    first = STANDARD(images, generator(7))
    second = STANDARD(images, generator(7))
    assert torch.allclose(first, second)
    assert not torch.allclose(STANDARD(images, generator(8)), first)


def test_flip_mirrors_some_images(images):
    out = random_horizontal_flip(images, generator(3))
    flipped = torch.flip(images, dims=[-1])
    matches_original = torch.tensor(
        [bool(torch.allclose(out[index], images[index])) for index in range(len(images))]
    )
    matches_flipped = torch.tensor(
        [bool(torch.allclose(out[index], flipped[index])) for index in range(len(images))]
    )
    assert bool((matches_original | matches_flipped).all())
    assert bool(matches_flipped.any())


def test_translation_moves_content_but_keeps_the_frame(images):
    assert torch.allclose(random_translate(images, 0, generator(1)), images)
    shifted = random_translate(images, 4, generator(1))
    assert shifted.shape == images.shape
    assert not torch.allclose(shifted, images)


def test_erasing_only_touches_a_patch(images):
    erased = random_erasing(images, probability=1.0, generator=generator(2), max_fraction=0.25)
    changed = (erased != images).float().mean(dim=(1, 2, 3))
    assert bool((changed > 0).all())
    assert float(changed.max()) < 0.30  # a quarter of the frame at most, plus rounding


def test_colour_jitter_leaves_the_tint_readable(images):
    """Brightness and contrast scale all channels together, so channel order survives."""
    jittered = colour_jitter(images, strength=0.2, generator=generator(4))
    before = images.mean(dim=(2, 3)).argmax(dim=1)
    after = jittered.mean(dim=(2, 3)).argmax(dim=1)
    assert float((before == after).float().mean()) > 0.9


def test_break_cue_destroys_the_channel_balance(images):
    broken = break_cue(images, strength=0.35, generator=generator(5))
    before = images.mean(dim=(2, 3))
    after = broken.mean(dim=(2, 3))
    before_spread = (before[:, 0] - before[:, 1]).std()
    after_spread = (after[:, 0] - after[:, 1]).std()
    assert float(after_spread) > float(before_spread)
    changed = (before.argmax(dim=1) != after.argmax(dim=1)).float().mean()
    assert float(changed) > 0.2  # the tint no longer identifies a class


def test_break_cue_keeps_the_shape_visible(images):
    """The intervention must remove the cue without removing the object."""
    broken = break_cue(images, strength=0.35, generator=generator(6))
    contrast_before = images.mean(dim=1).flatten(1).std(dim=1)
    contrast_after = broken.mean(dim=1).flatten(1).std(dim=1)
    assert float((contrast_after > 0.3 * contrast_before).float().mean()) > 0.8


def test_zero_strength_operations_are_identities(images):
    assert torch.allclose(colour_jitter(images, 0.0, generator(1)), images)
    assert torch.allclose(break_cue(images, 0.0, generator(1)), images)
    assert torch.allclose(random_erasing(images, 0.0, generator(1)), images)


def test_the_cue_breaking_preset_actually_breaks_the_cue():
    assert CUE_BREAKING.cue_break > 0.0
    assert STANDARD.cue_break == 0.0


def test_configuration_is_validated():
    with pytest.raises(ValueError, match="translate"):
        Augmentation(translate=-1).validate()
    with pytest.raises(ValueError, match="erase_probability"):
        Augmentation(erase_probability=1.5).validate()
    with pytest.raises(ValueError, match="destroys the shape"):
        Augmentation(cue_break=0.9).validate()
