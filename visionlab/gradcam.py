"""Grad-CAM, and two numbers that make it falsifiable.

A class activation map is usually presented as a heatmap next to an image, which is an
invitation to see what you expect.  Because this dataset ships the object's bounding box, the
map can be *scored* instead:

* **pointing game** - does the peak of the map fall inside the object box?  Reported against
  the mean box area share, which is where a random peak would land.  A pointing score at the
  baseline means the explanation carries no localisation information at all.
* **energy inside the box** - what share of the map's total mass lies on the object?  Less
  brittle than the peak, and it exposes a map that is diffuse rather than wrong.

Implementation notes that matter for correctness: gradients are taken with
``torch.autograd.grad`` against the captured activation instead of module backward hooks (which
have subtle semantics for modules with multiple inputs), the input is cloned with
``requires_grad_`` so the graph survives even when every parameter is frozen, and everything
runs inside ``torch.enable_grad()`` so the caller's ``no_grad`` context cannot silently produce
an all-zero map.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import CLASSES, Batch
from .models import target_layer as resolve_target_layer


class GradCAM:
    """Grad-CAM for one target layer.  Use as a context manager to remove the hook."""

    def __init__(self, model: nn.Module, layer: nn.Module | None = None) -> None:
        self.model = model
        self.layer = layer if layer is not None else resolve_target_layer(model)
        self._activations: torch.Tensor | None = None
        self._handle = self.layer.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output) -> None:
        self._activations = output

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_exc) -> None:
        self.remove()

    def __call__(
        self, images: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cams, logits)`` with cams shaped ``(B, H, W)`` and scaled to ``[0, 1]``."""
        if images.ndim != 4:
            raise ValueError("images must have shape (B, C, H, W)")
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.enable_grad():
                inputs = images.detach().clone().requires_grad_(True)
                logits = self.model(inputs)
                if self._activations is None:
                    raise RuntimeError(
                        "the target layer did not run; check that it belongs to this model"
                    )
                chosen = logits.argmax(dim=1) if targets is None else targets.to(logits.device)
                if chosen.shape != (logits.shape[0],):
                    raise ValueError("targets must hold one class index per image")
                score = logits.gather(1, chosen.view(-1, 1)).sum()
                gradients = torch.autograd.grad(score, self._activations)[0]
                activations = self._activations.detach()

            weights = gradients.detach().mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(
                cam, size=images.shape[-2:], mode="bilinear", align_corners=False
            )[:, 0]
            peak = cam.flatten(1).amax(dim=1).clamp(min=1e-12).view(-1, 1, 1)
            return (cam / peak).detach(), logits.detach()
        finally:
            self._activations = None
            if was_training:
                self.model.train()


def _as_numpy(cams: torch.Tensor) -> np.ndarray:
    return cams.detach().to("cpu").numpy()


def pointing_game(cams: torch.Tensor, boxes: np.ndarray) -> float:
    """Share of maps whose peak pixel falls inside the object box."""
    maps = _as_numpy(cams)
    if maps.shape[0] != boxes.shape[0]:
        raise ValueError("cams and boxes must align")
    hits = 0
    for index in range(maps.shape[0]):
        flat = int(np.argmax(maps[index]))
        y, x = divmod(flat, maps.shape[2])
        x0, y0, x1, y1 = (int(value) for value in boxes[index])
        hits += int(x0 <= x <= x1 and y0 <= y <= y1)
    return hits / maps.shape[0]


def mask_energy_inside(cams: torch.Tensor, boxes: np.ndarray) -> float:
    """Mean share of activation mass that lies on the object."""
    maps = _as_numpy(cams)
    if maps.shape[0] != boxes.shape[0]:
        raise ValueError("cams and boxes must align")
    shares = []
    for index in range(maps.shape[0]):
        single = maps[index]
        total = float(single.sum())
        if total <= 0:
            shares.append(0.0)
            continue
        x0, y0, x1, y1 = (int(value) for value in boxes[index])
        inside = float(single[y0 : y1 + 1, x0 : x1 + 1].sum())
        shares.append(inside / total)
    return float(np.mean(shares))


def box_area_share(boxes: np.ndarray, height: int, width: int) -> float:
    """Mean fraction of the frame covered by the object box - the pointing-game baseline."""
    widths = boxes[:, 2] - boxes[:, 0] + 1
    heights = boxes[:, 3] - boxes[:, 1] + 1
    return float(((widths * heights) / (height * width)).mean())


def compute_cams(
    model: nn.Module,
    batch: Batch,
    layer: nn.Module | None = None,
    device: str = "cpu",
    batch_size: int = 64,
    use_true_labels: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grad-CAM maps for a whole split, plus the logits that produced them.

    ``use_true_labels`` explains the *correct* class rather than the predicted one, which is the
    right question when asking "where is the evidence for the true label" - and it keeps the
    score comparable between a good model and a bad one.
    """
    images = torch.from_numpy(batch.images).to(device=device, dtype=torch.float32)
    labels = torch.from_numpy(batch.labels).to(device=device, dtype=torch.long)
    maps: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    with GradCAM(model, layer) as explainer:
        for start in range(0, images.shape[0], batch_size):
            chunk = images[start : start + batch_size]
            targets = labels[start : start + batch_size] if use_true_labels else None
            cam, logit = explainer(chunk, targets)
            maps.append(cam.to("cpu"))
            logits.append(logit.to("cpu"))
    return torch.cat(maps), torch.cat(logits)


def localization_report(
    model: nn.Module,
    batch: Batch,
    layer: nn.Module | None = None,
    device: str = "cpu",
    batch_size: int = 64,
    use_true_labels: bool = True,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Pointing game and mask energy, overall and per class, against the random baseline."""
    cams, logits = compute_cams(
        model, batch, layer, device, batch_size, use_true_labels=use_true_labels
    )
    height, width = batch.images.shape[-2:]
    baseline = box_area_share(batch.boxes, height, width)
    predictions = logits.argmax(dim=1).numpy()

    summary = {
        "split": batch.name,
        "images": len(batch),
        "pointing_game": round(pointing_game(cams, batch.boxes), 4),
        "random_baseline": round(baseline, 4),
        "energy_inside_box": round(mask_energy_inside(cams, batch.boxes), 4),
        "accuracy": round(float((predictions == batch.labels).mean()), 4),
    }

    rows = []
    for label, name in enumerate(CLASSES):
        mask = batch.labels == label
        if not bool(mask.any()):
            continue
        rows.append(
            {
                "class": name,
                "images": int(mask.sum()),
                "pointing_game": round(pointing_game(cams[mask], batch.boxes[mask]), 4),
                "energy_inside_box": round(mask_energy_inside(cams[mask], batch.boxes[mask]), 4),
                "accuracy": round(float((predictions[mask] == batch.labels[mask]).mean()), 4),
            }
        )
    return summary, pd.DataFrame(rows)
