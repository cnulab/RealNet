import copy

from torch import nn


def init_weights_normal(module, std=0.01):
    for m in module.modules():
        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.Linear)
            or isinstance(m, nn.ConvTranspose2d)
        ):
            nn.init.normal_(m.weight.data, std=std)
            if m.bias is not None:
                m.bias.data.zero_()


def init_weights_xavier(module, method):
    for m in module.modules():
        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.Linear)
            or isinstance(m, nn.ConvTranspose2d)
        ):
            if "normal" in method:
                nn.init.xavier_normal_(m.weight.data)
            elif "uniform" in method:
                nn.init.xavier_uniform_(m.weight.data)
            else:
                raise NotImplementedError(f"{method} not supported")
            if m.bias is not None:
                m.bias.data.zero_()


def init_weights_msra(module, method):
    for m in module.modules():
        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.Linear)
            or isinstance(m, nn.ConvTranspose2d)
        ):
            if "normal" in method:
                nn.init.kaiming_normal_(m.weight.data, a=1)
            elif "uniform" in method:
                nn.init.kaiming_uniform_(m.weight.data, a=1)
            else:
                raise NotImplementedError(f"{method} not supported")
            if m.bias is not None:
                m.bias.data.zero_()


def initialize(model, method, **kwargs):
    # initialize BN, Conv, & FC with different methods
    # initialize BN
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    # initialize Conv & FC
    if method == "normal":
        init_weights_normal(model, **kwargs)
    elif "msra" in method:
        init_weights_msra(model, method)
    elif "xavier" in method:
        init_weights_xavier(model, method)
    else:
        raise NotImplementedError(f"{method} not supported")


def initialize_from_cfg(model, cfg):
    if cfg is None:
        initialize(model, "normal", std=0.01)
        return

    cfg = copy.deepcopy(cfg)
    method = cfg.pop("method")
    initialize(model, method, **cfg)
