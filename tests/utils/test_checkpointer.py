import argparse
import json
import shutil

import torch
from pytorch_lightning import Trainer
from solo.methods import BarlowTwins
from solo.utils.checkpointer import Checkpointer

from ..methods.utils import DATA_KWARGS, gen_base_kwargs, prepare_dummy_dataloaders


def test_checkpointer():
    method_kwargs = {
        "name": "barlow_twins",
        "proj_hidden_dim": 2048,
        "output_dim": 2048,
        "lamb": 5e-3,
        "scale_loss": 0.025,
    }

    # normal training
    BASE_KWARGS = gen_base_kwargs(cifar=False, multicrop=False)
    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
    model = BarlowTwins(**kwargs)

    args = argparse.Namespace(**kwargs)

    # checkpointer
    ckpt_callback = Checkpointer(args)

    trainer = Trainer.from_argparse_args(
        args,
        checkpoint_callback=False,
        limit_train_batches=2,
        limit_val_batches=2,
        callbacks=[ckpt_callback],
    )

    train_dl, val_dl = prepare_dummy_dataloaders(
        "imagenet100",
        n_crops=BASE_KWARGS["n_crops"],
        n_small_crops=0,
        n_classes=BASE_KWARGS["n_classes"],
        multicrop=False,
    )

    trainer.fit(model, train_dl, val_dl)

    # check if checkpointer dumped the args
    args_path = ckpt_callback.path / "args.json"
    assert args_path.exists()

    # check if the args are correct
    loaded_args = json.load(open(args_path))
    assert loaded_args == vars(args)

    # check if checkpointer dumped the checkpoint
    ckpt_path = ckpt_callback.path / ckpt_callback.ckpt_placeholder.format(trainer.current_epoch)

    assert ckpt_path.exists()

    # check if the checkpoint contains the correct keys
    ckpt = torch.load(ckpt_path)
    expected_keys = [
        "epoch",
        "global_step",
        "pytorch-lightning_version",
        "state_dict",
        "callbacks",
        "optimizer_states",
        "lr_schedulers",
    ]
    assert list(ckpt.keys()) == expected_keys

    parser = argparse.ArgumentParser()
    ckpt_callback.add_checkpointer_args(parser)
    args = [vars(action)["dest"] for action in vars(parser)["_actions"]]
    assert "checkpoint_dir" in args
    assert "checkpoint_frequency" in args

    # clean stuff
    shutil.rmtree(ckpt_callback.logdir)
