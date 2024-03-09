import copy
import importlib
import torch
import torch.nn as nn
from utils.misc_helper import to_device


class ModelHelper(nn.Module):
    """Build model from cfg"""

    def __init__(self, cfg):
        super(ModelHelper, self).__init__()

        self.frozen_layers = []

        for cfg_subnet in cfg:
            mname = cfg_subnet["name"]
            kwargs=cfg_subnet.get('kwargs',{})
            mtype = cfg_subnet["type"]

            if cfg_subnet.get("frozen", False):
                self.frozen_layers.append(mname)

            if cfg_subnet.get("prev", None) is not None:
                prev_module = getattr(self, cfg_subnet["prev"])
                kwargs["inplanes"] = prev_module.get_outplanes()
                kwargs["instrides"] = prev_module.get_outstrides()

            module = self.build(mtype, kwargs)
            self.add_module(mname, module)

    def build(self, mtype, kwargs):
        module_name, cls_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        return cls(**kwargs)

    def cuda(self):
        self.device = torch.device("cuda")
        return super(ModelHelper, self).cuda()


    def cpu(self):
        self.device = torch.device("cpu")
        return super(ModelHelper, self).cpu()

    def forward(self, input, train=False):
        input = copy.copy(input)
        if input["image"].device != self.device:
            input = to_device(input, device=self.device)
        for submodule in self.children():
            output = submodule(input,train)
            input.update(output)
        return input

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self
