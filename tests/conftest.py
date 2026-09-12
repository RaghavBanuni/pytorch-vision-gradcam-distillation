"""Shared fixtures.

Small images, a narrow network and a handful of epochs: enough for every behavioural claim in
the suite to be checked on CPU in well under a minute.  The training fixture is session-scoped
because several files need a model that has actually learned something.
"""

from __future__ import annotations

import pytest

from visionlab.augment import STANDARD
from visionlab.data import ImageConfig, build_dataset
from visionlab.models import SmallResNet
from visionlab.train import TrainConfig, Trainer

CONFIG = ImageConfig(size=32, n_train=384, n_val=96, n_test=96, seed=5)


@pytest.fixture(scope="session")
def config() -> ImageConfig:
    return CONFIG


@pytest.fixture(scope="session")
def datasets():
    return build_dataset(CONFIG)


@pytest.fixture(scope="session")
def trained(datasets):
    model = SmallResNet(n_classes=4, width=8, blocks=(2, 2))
    history = Trainer(
        TrainConfig(epochs=4, batch_size=32, lr=5e-3, seed=0, patience=4)
    ).fit(model, datasets["train"], datasets["val"], STANDARD)
    return model, history
