import random

import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from walrus.train import CONFIG_DIR, CONFIG_NAME
from walrus.utils.experiment_utils import seed_all


def _all_params(cfg_overrides, seed):
    """Build a model under a given seed and flatten every parameter into one vector.

    Not just the first parameter: `IsotropicModel.encoder_dummy` is `torch.ones(1)`, so
    it is identical under any seed and would make the comparison vacuous.
    """
    import os.path as osp
    import pathlib

    cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)
    with initialize(config_path=str(cfg_dir)):
        cfg = compose(config_name=CONFIG_NAME, overrides=cfg_overrides)
        seed_all(seed)
        model = instantiate(cfg.model, n_states=8)
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_seed_all_covers_every_rng():
    """torch, numpy and python random are all drawn from in the training path."""
    seed_all(1234)
    draws = (torch.randn(3), np.random.rand(3), [random.random() for _ in range(3)])
    seed_all(1234)
    again = (torch.randn(3), np.random.rand(3), [random.random() for _ in range(3)])

    assert torch.equal(draws[0], again[0])
    assert np.array_equal(draws[1], again[1])
    assert draws[2] == again[2]


def test_config_exposes_a_seed(default_conf):
    overrides = [f"{key}={val}" for key, val in default_conf.items()]
    import os.path as osp
    import pathlib

    cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)
    with initialize(config_path=str(cfg_dir)):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
    assert "seed" in cfg


def test_same_seed_gives_identical_weights(default_conf):
    """Weight init must be a pure function of the seed.

    This is what makes the noise-floor runs in SCALING_STRATEGY.md 5.4 interpretable, and
    it is also load-bearing across ranks: `distribute_model` builds FSDP without
    `sync_module_states`, so ranks that initialise differently never reconcile.
    """
    overrides = [f"{key}={val}" for key, val in default_conf.items()]
    first = _all_params(overrides, seed=0)
    same = _all_params(overrides, seed=0)
    different = _all_params(overrides, seed=1)

    assert torch.equal(first, same)
    assert not torch.equal(first, different)
