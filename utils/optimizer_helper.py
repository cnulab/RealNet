import torch


def get_optimizer(parameters, config):
    if config.type == "AdamW":
        return torch.optim.AdamW(parameters, **config.kwargs)
    elif config.type == "Adam":
        return torch.optim.Adam(parameters, **config.kwargs)
    elif config.type == "SGD":
        return torch.optim.SGD(parameters, **config.kwargs)
    else:
        raise NotImplementedError
