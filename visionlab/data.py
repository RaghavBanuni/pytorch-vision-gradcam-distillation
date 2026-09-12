"""A procedural image dataset with a planted shortcut and known object boxes.

Real image datasets cannot answer the two questions this project is about - *where is the
object* and *is there a spurious cue* - so the data is generated, and both answers come with
it.

Each image contains one shape (circle, square, triangle, cross) at a random position, size
and rotation, drawn in a colour picked independently of the label so that colour carries no
information.  The label is the shape.

The background carries a **cue**: a tint that identifies a class.  How often that tint agrees
with the true label is controlled by ``cue_strength``:

* ``aligned``  - the tint matches the label with probability ``cue_strength``.  Training and
  the i.i.d. test split use this, which is exactly why an i.i.d. test score cannot detect the
  shortcut: the shortcut is in the test set too.
* ``random``   - the tint is drawn uniformly, so it carries no information.
* ``inverted`` - the tint systematically names the *wrong* class.  A model that learned the
  shape is unaffected; a model that learned the tint collapses below chance.

The three test regimes are **paired**: they are rendered from the same :class:`Latent` values -
same shapes, positions, rotations, colours and even the same noise draw - and differ only in
the background tint.  That is deliberate.  If each regime were sampled independently, an
accuracy gap between them would confound the cue with ordinary sampling noise, and the whole
measurement would be weaker than it looks.

Every sample also carries the tight bounding box of its shape, which is what makes the
Grad-CAM pointing game possible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASSES: tuple[str, ...] = ("circle", "square", "triangle", "cross")
CUE_MODES: tuple[str, ...] = ("aligned", "random", "inverted")

# one tint per class; deliberately easy to learn, which is the point of a shortcut
CUE_TINTS: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.7, 0.7, 0.0],
    ],
    dtype=np.float32,
)

# foreground palette, sampled independently of the label so colour is not informative
PALETTE: np.ndarray = np.array(
    [
        [0.95, 0.95, 0.95],
        [0.08, 0.08, 0.08],
        [0.90, 0.55, 0.10],
        [0.15, 0.45, 0.85],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class ImageConfig:
    """Generator settings.  Defaults train on CPU in a couple of minutes."""

    size: int = 48
    n_train: int = 2_400
    n_val: int = 600
    n_test: int = 600
    cue_strength: float = 0.95
    cue_contrast: float = 0.22
    background_level: float = 0.45
    noise_std: float = 0.05
    min_object: float = 0.28
    max_object: float = 0.60
    seed: int = 17

    def validate(self) -> None:
        if self.size < 16:
            raise ValueError("size must be at least 16 pixels for the shapes to be legible")
        if min(self.n_train, self.n_val, self.n_test) < 16:
            raise ValueError("each split needs at least 16 images")
        if not 0.0 <= self.cue_strength <= 1.0:
            raise ValueError("cue_strength must lie in [0, 1]")
        if not 0.0 < self.cue_contrast < 0.5:
            raise ValueError("cue_contrast must lie in (0, 0.5) to stay inside the pixel range")
        if not 0.0 < self.min_object < self.max_object <= 0.9:
            raise ValueError("object size fractions must satisfy 0 < min < max <= 0.9")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")


@dataclass(frozen=True)
class Latent:
    """Everything about one image *except* the background cue.

    Keeping the latent state separate from the cue is what makes the three test regimes a
    paired comparison: the same latents are rendered three times under three tints, so nothing
    but the tint can explain a difference in accuracy.  ``noise_seed`` is stored rather than
    the noise itself so the pairing survives without holding arrays in memory.
    """

    label: int
    radius: float
    centre: tuple[float, float]
    angle: float
    colour: int
    noise_seed: int


@dataclass(frozen=True)
class Batch:
    """One split: images plus every label the harness needs.

    ``boxes`` are inclusive pixel bounds ``(x0, y0, x1, y1)``; ``cue_labels`` records which
    class the background tint names, so cue agreement can be measured rather than assumed.
    """

    images: np.ndarray  # (N, 3, H, W) float32 in [0, 1]
    labels: np.ndarray  # (N,) int64
    boxes: np.ndarray  # (N, 4) int64
    cue_labels: np.ndarray  # (N,) int64
    name: str

    def __post_init__(self) -> None:
        if self.images.ndim != 4 or self.images.shape[1] != 3:
            raise ValueError("images must have shape (N, 3, H, W)")
        count = self.images.shape[0]
        if not (len(self.labels) == len(self.cue_labels) == len(self.boxes) == count):
            raise ValueError("labels, cue labels and boxes must align with the images")
        if self.boxes.shape[1] != 4:
            raise ValueError("boxes must have four columns")

    def __len__(self) -> int:
        return int(self.images.shape[0])

    @property
    def cue_agreement(self) -> float:
        """Share of images whose background tint names the true class."""
        return float((self.labels == self.cue_labels).mean())


def _rotate(y: np.ndarray, x: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    cos, sin = np.cos(angle), np.sin(angle)
    return y * cos - x * sin, y * sin + x * cos


def shape_mask(kind: str, size: int, radius: float, centre, angle: float) -> np.ndarray:
    """Boolean mask of one shape, drawn analytically so edges stay clean at 48 pixels.

    ``centre`` is ``(row, column)``.  Every shape stays within ``radius`` of the centre under
    any rotation, which is what lets the caller guarantee the shape is inside the frame.
    """
    if kind not in CLASSES:
        raise ValueError(f"unknown shape {kind!r}; expected one of {CLASSES}")
    if radius <= 0:
        raise ValueError("radius must be positive")
    grid = np.arange(size, dtype=np.float32)
    yy, xx = np.meshgrid(grid - centre[0], grid - centre[1], indexing="ij")
    yy, xx = _rotate(yy, xx, angle)

    if kind == "circle":
        return (yy**2 + xx**2) <= radius**2
    if kind == "square":
        return (np.abs(yy) <= radius * 0.75) & (np.abs(xx) <= radius * 0.75)
    if kind == "cross":
        arm = radius * 0.32
        return ((np.abs(xx) <= arm) & (np.abs(yy) <= radius)) | (
            (np.abs(yy) <= arm) & (np.abs(xx) <= radius)
        )

    # equilateral triangle with vertices at (0, -r), (0.866r, 0.5r), (-0.866r, 0.5r) in (x, y).
    # A point is inside when all three edge functions share a sign; with this vertex order the
    # interior is the all-positive region (verified at the centroid, where every edge gives
    # +0.866 r^2).
    half_base = 0.866 * radius
    edge_a = half_base * (yy + radius) - 1.5 * radius * xx
    edge_b = -2.0 * half_base * (yy - 0.5 * radius)
    edge_c = half_base * (yy - 0.5 * radius) + 1.5 * radius * (xx + half_base)
    return (edge_a >= 0) & (edge_b >= 0) & (edge_c >= 0)


def _bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        raise ValueError("the shape fell entirely outside the frame")
    return int(columns[0]), int(rows[0]), int(columns[-1]), int(rows[-1])


def sample_latents(count: int, config: ImageConfig, seed: int) -> list[Latent]:
    """Sample the object state for one split, with classes balanced by construction.

    The margin keeps every shape fully inside the frame, so a bounding box is always the true
    extent of the object rather than a clipped guess.
    """
    config.validate()
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    labels = np.tile(np.arange(len(CLASSES)), count // len(CLASSES) + 1)[:count]
    rng.shuffle(labels)

    size = config.size
    latents: list[Latent] = []
    for label in labels:
        radius = float(rng.uniform(config.min_object, config.max_object)) * size / 2.0
        margin = radius * 1.05 + 1.0
        latents.append(
            Latent(
                label=int(label),
                radius=radius,
                centre=(
                    float(rng.uniform(margin, size - margin)),
                    float(rng.uniform(margin, size - margin)),
                ),
                angle=float(rng.uniform(0.0, 2.0 * np.pi)),
                colour=int(rng.integers(len(PALETTE))),
                noise_seed=int(rng.integers(0, 2**31 - 1)),
            )
        )
    return latents


def assign_cues(
    labels: np.ndarray, cue_mode: str, config: ImageConfig, seed: int
) -> np.ndarray:
    """Which class the background tint names, per image.

    In ``aligned`` mode a disagreeing tint is drawn uniformly from the *other* classes, so the
    realised agreement rate is ``cue_strength`` rather than ``cue_strength + noise``.
    """
    if cue_mode not in CUE_MODES:
        raise ValueError(f"unknown cue mode {cue_mode!r}; expected one of {CUE_MODES}")
    labels = np.asarray(labels, dtype=np.int64)
    n_classes = len(CLASSES)
    if cue_mode == "inverted":
        return ((labels + 1) % n_classes).astype(np.int64)

    rng = np.random.default_rng(seed)
    if cue_mode == "random":
        return rng.integers(n_classes, size=len(labels)).astype(np.int64)
    keep = rng.random(len(labels)) < config.cue_strength
    offsets = rng.integers(1, n_classes, size=len(labels))
    return np.where(keep, labels, (labels + offsets) % n_classes).astype(np.int64)


def render(
    latent: Latent, cue_label: int, config: ImageConfig
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """One image: tinted background, one shape, gaussian noise.

    Pure in ``(latent, cue_label, config)`` - no shared random state - which is what lets two
    regimes render byte-identical images whenever their tints happen to agree.
    """
    if not 0 <= cue_label < len(CLASSES):
        raise ValueError(f"cue_label must index a class in [0, {len(CLASSES)})")
    size = config.size
    background = np.full(
        (3, size, size), config.background_level, dtype=np.float32
    ) + config.cue_contrast * CUE_TINTS[cue_label][:, None, None]

    mask = shape_mask(CLASSES[latent.label], size, latent.radius, latent.centre, latent.angle)
    if not mask.any():  # pragma: no cover - the margin makes this unreachable
        raise ValueError("empty shape mask")

    colour = PALETTE[latent.colour]
    # keep the shape visible against the tinted background whatever colour was drawn
    if abs(float(colour.mean()) - config.background_level) < 0.25:
        colour = PALETTE[0] if config.background_level < 0.5 else PALETTE[1]

    image = background.copy()
    image[:, mask] = colour[:, None]
    if config.noise_std > 0:
        noise = np.random.default_rng(latent.noise_seed).normal(
            0.0, config.noise_std, size=image.shape
        )
        image = image + noise.astype(np.float32)
    return np.clip(image, 0.0, 1.0).astype(np.float32), _bounding_box(mask)


def generate(
    count: int,
    cue_mode: str,
    config: ImageConfig,
    seed: int,
    name: str = "split",
    latents: list[Latent] | None = None,
) -> Batch:
    """Generate one split.  Pass ``latents`` to re-render an existing split under a new cue."""
    config.validate()
    if cue_mode not in CUE_MODES:
        raise ValueError(f"unknown cue mode {cue_mode!r}; expected one of {CUE_MODES}")
    if count < 1:
        raise ValueError("count must be positive")
    if latents is None:
        latents = sample_latents(count, config, seed)
    elif len(latents) != count:
        raise ValueError("the supplied latents do not match the requested count")

    labels = np.array([latent.label for latent in latents], dtype=np.int64)
    # a separate stream for the cue: paired splits share latents but must not share tints
    cues = assign_cues(labels, cue_mode, config, seed + 9_973)

    images = np.empty((count, 3, config.size, config.size), dtype=np.float32)
    boxes = np.empty((count, 4), dtype=np.int64)
    for index, latent in enumerate(latents):
        images[index], box = render(latent, int(cues[index]), config)
        boxes[index] = box
    return Batch(images=images, labels=labels, boxes=boxes, cue_labels=cues, name=name)


def build_dataset(config: ImageConfig | None = None) -> dict[str, Batch]:
    """Train, validation and the three test regimes.

    Train, val and test draw independent latents, so no image can appear in two of them.  The
    three *test* regimes deliberately share one set of latents and differ only in the
    background cue: any accuracy gap between them is caused by the cue and nothing else.
    """
    settings = config or ImageConfig()
    settings.validate()
    base = settings.seed
    test_latents = sample_latents(settings.n_test, settings, base + 2)
    return {
        "train": generate(settings.n_train, "aligned", settings, base, "train"),
        "val": generate(settings.n_val, "aligned", settings, base + 1, "val"),
        "test": generate(
            settings.n_test, "aligned", settings, base + 2, "test", test_latents
        ),
        "test_cue_broken": generate(
            settings.n_test, "random", settings, base + 3, "test_cue_broken", test_latents
        ),
        "test_cue_inverted": generate(
            settings.n_test, "inverted", settings, base + 4, "test_cue_inverted", test_latents
        ),
    }


def describe(batch: Batch) -> dict[str, float]:
    """Summary of one split, including how much the cue could be exploited."""
    widths = batch.boxes[:, 2] - batch.boxes[:, 0] + 1
    heights = batch.boxes[:, 3] - batch.boxes[:, 1] + 1
    frame = batch.images.shape[2] * batch.images.shape[3]
    smallest_class = int(np.bincount(batch.labels, minlength=len(CLASSES)).min())
    return {
        "split": batch.name,
        "images": len(batch),
        "cue_agreement": round(batch.cue_agreement, 4),
        "mean_box_share": round(float(((widths * heights) / frame).mean()), 4),
        "class_balance": round(smallest_class / len(batch), 4),
        "pixel_mean": round(float(batch.images.mean()), 4),
    }
