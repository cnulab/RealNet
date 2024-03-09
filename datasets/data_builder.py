import copy
import logging

from datasets.sdas_dataset import build_sdas_dataloader
from datasets.realnet_dataset import build_realnet_dataloader


def build(cfg, training, distributed=True):

    if training:
        cfg.update(cfg.get("train", {}))
    else:
        cfg.update(cfg.get("test", {}))

    type= cfg["type"]
    if type == "sdas":
        data_loader = build_sdas_dataloader(cfg, training, distributed)
    elif type in ['mvtec','visa','mpdd','btad']:
        data_loader = build_realnet_dataloader(cfg, training, distributed)
    else:
        raise NotImplementedError(f"dataset type '{type}' is not supported")
    return data_loader


def build_dataloader(cfg_dataset, distributed=True):

    train_loader = None
    if cfg_dataset.get("train", None):
        train_loader = build(copy.deepcopy(cfg_dataset), training=True, distributed=distributed)

    test_loader = None
    if cfg_dataset.get("test", None):
        test_loader = build(copy.deepcopy(cfg_dataset), training=False, distributed=distributed)

    return train_loader, test_loader
