"""Tests for the generator.

The properties the whole project rests on are asserted here: the cue really is correlated with
the label in the training regime (and really is inverted in the OOD regime), the three test
regimes really are paired so that only the cue differs, and the bounding box really does
contain the object - otherwise the Grad-CAM pointing score would be measuring nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionlab.data import (
    CLASSES,
    ImageConfig,
    build_dataset,
    describe,
    generate,
    render,
    sample_latents,
    shape_mask,
)


def test_generation_is_reproducible(config):
    first = generate(64, "aligned", config, seed=3)
    second = generate(64, "aligned", config, seed=3)
    assert np.array_equal(first.images, second.images)
    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.boxes, second.boxes)
    assert np.array_equal(first.cue_labels, second.cue_labels)


def test_rendering_is_a_pure_function_of_the_latent(config):
    """No shared random state - this is what makes the paired regimes possible."""
    latent = sample_latents(16, config, seed=1)[0]
    first, box_a = render(latent, 2, config)
    second, box_b = render(latent, 2, config)
    assert np.array_equal(first, second)
    assert box_a == box_b
    tinted, box_c = render(latent, 3, config)
    assert not np.array_equal(first, tinted)
    assert box_c == box_a  # the object did not move, only the background changed


def test_images_are_valid_tensors(datasets, config):
    for batch in datasets.values():
        assert batch.images.dtype == np.float32
        assert batch.images.shape[1:] == (3, config.size, config.size)
        assert batch.images.min() >= 0.0 and batch.images.max() <= 1.0
        assert set(np.unique(batch.labels)) <= set(range(len(CLASSES)))


def test_classes_are_balanced(datasets):
    for batch in datasets.values():
        counts = np.bincount(batch.labels, minlength=len(CLASSES))
        assert counts.max() - counts.min() <= 1


def test_the_cue_is_correlated_in_the_training_regime(datasets, config):
    assert datasets["train"].cue_agreement == pytest.approx(config.cue_strength, abs=0.08)
    assert datasets["test"].cue_agreement == pytest.approx(config.cue_strength, abs=0.10)


def test_the_cue_is_uninformative_when_broken(datasets):
    assert datasets["test_cue_broken"].cue_agreement == pytest.approx(0.25, abs=0.12)


def test_the_cue_is_systematically_wrong_when_inverted(datasets):
    """Zero agreement is what makes the collapse of a shortcut model unambiguous."""
    assert datasets["test_cue_inverted"].cue_agreement == 0.0


def test_a_disagreeing_tint_is_never_the_true_label(datasets):
    """In aligned mode the realised agreement must equal cue_strength, not exceed it."""
    batch = datasets["train"]
    disagreeing = batch.cue_labels[batch.cue_labels != batch.labels]
    assert len(disagreeing) > 0
    assert len(np.unique(disagreeing)) > 1  # spread over the other classes, not a single one


def test_boxes_lie_inside_the_frame(datasets, config):
    for batch in datasets.values():
        x0, y0, x1, y1 = (batch.boxes[:, index] for index in range(4))
        assert (x0 >= 0).all() and (y0 >= 0).all()
        assert (x1 < config.size).all() and (y1 < config.size).all()
        assert (x1 >= x0).all() and (y1 >= y0).all()


def test_the_object_is_actually_inside_its_box(datasets):
    """The pixel furthest from the background colour must fall inside the box.

    Shape contrast is at least 0.25 while the noise standard deviation is 0.05, so this is a
    safe check - and it is the assumption the pointing game depends on.
    """
    batch = datasets["test"]
    for index in range(0, len(batch), 7):
        image = batch.images[index].mean(axis=0)
        deviation = np.abs(image - np.median(image))
        y, x = np.unravel_index(int(np.argmax(deviation)), image.shape)
        x0, y0, x1, y1 = batch.boxes[index]
        assert x0 <= x <= x1 and y0 <= y <= y1, index


def test_train_val_and_test_share_no_images(datasets):
    seen: set[bytes] = set()
    for name in ("train", "val", "test"):
        batch = datasets[name]
        digests = {batch.images[index].tobytes() for index in range(len(batch))}
        assert not (digests & seen), name
        seen |= digests


def test_the_three_test_regimes_are_paired(datasets):
    """Same objects, same noise; only the tint differs.

    Where two regimes happen to draw the same tint the images must be byte-identical, and where
    the tints differ the images must differ.  Without this pairing, an accuracy gap between the
    regimes would confound the cue with ordinary sampling noise.
    """
    reference = datasets["test"]
    for name in ("test_cue_broken", "test_cue_inverted"):
        other = datasets[name]
        assert np.array_equal(other.labels, reference.labels)
        assert np.array_equal(other.boxes, reference.boxes)
        same_tint = other.cue_labels == reference.cue_labels
        assert same_tint.any() or name == "test_cue_inverted"
        for index in np.flatnonzero(same_tint)[:5]:
            assert np.array_equal(other.images[index], reference.images[index])
        for index in np.flatnonzero(~same_tint)[:5]:
            assert not np.array_equal(other.images[index], reference.images[index])


def test_supplied_latents_must_match_the_count(config):
    latents = sample_latents(32, config, seed=2)
    with pytest.raises(ValueError, match="do not match the requested count"):
        generate(16, "aligned", config, seed=2, latents=latents)


@pytest.mark.parametrize(
    ("kind", "expected_area_factor"),
    [
        ("circle", np.pi),
        ("square", 2.25),
        ("triangle", 3 * np.sqrt(3) / 4),
        ("cross", 2.15),
    ],
)
def test_shape_masks_have_the_right_area(kind, expected_area_factor):
    """Each analytic mask is checked against its closed-form area in units of r^2."""
    radius = 12.0
    mask = shape_mask(kind, 64, radius, (32.0, 32.0), angle=0.0)
    area = mask.sum() / radius**2
    assert area == pytest.approx(expected_area_factor, rel=0.12), kind


def test_shapes_stay_within_their_radius():
    radius = 10.0
    centre = (32.0, 32.0)
    for kind in CLASSES:
        mask = shape_mask(kind, 64, radius, centre, angle=0.7)
        rows, columns = np.nonzero(mask)
        distance = np.sqrt((rows - centre[0]) ** 2 + (columns - centre[1]) ** 2)
        assert distance.max() <= radius + 1.5, kind


def test_rotation_changes_a_shape_but_not_a_circle():
    upright = shape_mask("square", 64, 12.0, (32.0, 32.0), angle=0.0)
    turned = shape_mask("square", 64, 12.0, (32.0, 32.0), angle=0.6)
    assert not np.array_equal(upright, turned)
    assert turned.sum() == pytest.approx(upright.sum(), rel=0.15)

    circle_a = shape_mask("circle", 64, 12.0, (32.0, 32.0), angle=0.0)
    circle_b = shape_mask("circle", 64, 12.0, (32.0, 32.0), angle=1.1)
    assert np.array_equal(circle_a, circle_b)


def test_shape_mask_rejects_bad_input():
    with pytest.raises(ValueError, match="unknown shape"):
        shape_mask("hexagon", 32, 5.0, (16.0, 16.0), 0.0)
    with pytest.raises(ValueError, match="radius must be positive"):
        shape_mask("circle", 32, 0.0, (16.0, 16.0), 0.0)


def test_describe_reports_the_cue_and_the_object_size(datasets):
    summary = describe(datasets["train"])
    assert summary["images"] == len(datasets["train"])
    assert 0.0 < summary["mean_box_share"] < 1.0
    assert summary["class_balance"] > 0.2


def test_configuration_is_validated():
    with pytest.raises(ValueError, match="at least 16 pixels"):
        build_dataset(ImageConfig(size=8))
    with pytest.raises(ValueError, match=r"cue_strength"):
        build_dataset(ImageConfig(cue_strength=1.5))
    with pytest.raises(ValueError, match="object size fractions"):
        build_dataset(ImageConfig(min_object=0.7, max_object=0.5))


def test_unknown_cue_mode_is_rejected(config):
    with pytest.raises(ValueError, match="unknown cue mode"):
        generate(32, "shuffled", config, seed=1)


def test_render_rejects_a_cue_outside_the_class_range(config):
    latent = sample_latents(16, config, seed=4)[0]
    with pytest.raises(ValueError, match="cue_label must index a class"):
        render(latent, len(CLASSES), config)
