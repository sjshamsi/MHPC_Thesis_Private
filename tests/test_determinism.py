"""A run's random draws are a function of (seed, rank, epoch): resuming changes nothing."""

import os.path as osp
import pathlib
import shutil

import torch
from hydra import compose, initialize

from walrus.train import CONFIG_DIR, CONFIG_NAME, main


def _train(default_conf, save_dir, name, max_epoch, seed=0):
    overrides = {**default_conf, "checkpoint.save_dir": save_dir, "name": name}
    overrides = [f"{k}={v}" for k, v in overrides.items()] + [
        "+model.include_d=[2]",
        "checkpoint.checkpoint_frequency=1",
        f"trainer.max_epoch={max_epoch}",
        f"seed={seed}",
    ]
    cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)
    with initialize(config_path=str(cfg_dir)):
        main(compose(config_name=CONFIG_NAME, overrides=overrides))


def _weights(step_dir: pathlib.Path):
    # Local (non-distributed) runs save one torch file instead of a DCP directory.
    return torch.load(step_dir / "full_checkpoint.pt", weights_only=False)["app"]["model"]


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_resume_equals_uninterrupted(default_conf, tmp_path):
    _train(default_conf, tmp_path / "a", "a", max_epoch=2)
    # A job killed after saving epoch 1: same config (max_epoch sets the LR schedule), but
    # only step_1 survives, so the next job auto-resumes from it and re-runs epoch 2.
    _train(default_conf, tmp_path / "b", "b", max_epoch=2)
    first_try = _weights(tmp_path / "b" / "step_2")
    shutil.rmtree(tmp_path / "b" / "step_2")
    for link in ("last", "best"):
        (tmp_path / "b" / link).unlink()
        (tmp_path / "b" / link).symlink_to("step_1")
    _train(default_conf, tmp_path / "b", "b", max_epoch=2)
    resumed = _weights(tmp_path / "b" / "step_2")
    assert _same(first_try, resumed)
    assert _same(_weights(tmp_path / "a" / "step_2"), resumed)


def test_seed_controls_the_run(default_conf, tmp_path):
    _train(default_conf, tmp_path / "a", "a", max_epoch=2, seed=0)
    _train(default_conf, tmp_path / "b", "b", max_epoch=2, seed=0)
    _train(default_conf, tmp_path / "c", "c", max_epoch=2, seed=1)
    a, b, c = (_weights(tmp_path / d / "step_2") for d in "abc")
    assert _same(a, b)
    assert not _same(a, c)
