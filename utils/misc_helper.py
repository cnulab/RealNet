
import logging
from datetime import datetime
import random
import os
import torch
import numpy as np
from collections.abc import Mapping
import shutil


def map_func(storage, location):
    return storage.cuda()


def load_pertrain_weights(pertrain_path,model):

    pertrain_state_dict = torch.load(pertrain_path,map_location=map_func)

    if 'state_dict' in list(pertrain_state_dict.keys()):
        pertrain_state_dict = pertrain_state_dict['state_dict']
        pertrain_state_dict_= {}
        for key in pertrain_state_dict:
            if key.startswith("module"):
                key_new=key.replace("module.","")
            else:
                key_new=key
            pertrain_state_dict_[key_new]=pertrain_state_dict[key]
        pertrain_state_dict=pertrain_state_dict_

    model_state_dict = model.state_dict()

    match_keys = {}
    unexpected_keys = []

    for key in pertrain_state_dict:
        if key in model_state_dict.keys():
            if pertrain_state_dict[key].size() == model_state_dict[key].size():
                match_keys[key] = pertrain_state_dict[key]
        else:
            unexpected_keys.append(key)

    missing_keys = [key for key in model_state_dict.keys() if key not in match_keys]

    for key in match_keys:
        model_state_dict[key].data.copy_(match_keys[key])

    return missing_keys,unexpected_keys




def summary_model(model,logger):
    logger.info("************* model summary *************")
    for name,child in model.named_children():
        logger.info('{}: {} param'.format(name,sum([param.numel()for param in child.parameters()])))
    total_num=sum([param.numel() for param in model.parameters()])
    frozen_num = sum([param.numel()  for layer in model.frozen_layers for param in getattr(model,layer).parameters()])
    logger.info("************* model summary *************")
    logger.info('total: {} param'.format(total_num))
    logger.info('frozen: {} param'.format(frozen_num))
    logger.info('trainable: {} param'.format(total_num-frozen_num))
    logger.info("************* model summary *************")



def create_logger(name, log_file, level=logging.INFO):
    log = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s"
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    log.setLevel(level)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


def get_current_time():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    return current_time


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()

    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


def to_device(input, device="cuda", dtype=None):
    """Transfer data between devidces"""

    if "image" in input:
        input["image"] = input["image"].to(dtype=dtype)

    def transfer(x):
        if torch.is_tensor(x):
            return x.to(device=device)
        elif isinstance(x, list):
            return [transfer(_) for _ in x]
        elif isinstance(x, Mapping):
            return type(x)({k: transfer(v) for k, v in x.items()})
        else:
            return x

    return {k: transfer(v) for k, v in input.items()}



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count



def save_checkpoint(state, config, class_name=None,epoch=None):
    if class_name:
        folder = os.path.join(config.checkpoints_path,class_name)
    else:
        folder = os.path.join(config.checkpoints_path)

    os.makedirs(folder,exist_ok=True)
    if epoch:
        torch.save(state, os.path.join(folder, f"ckpt_{epoch}.pth.tar"))
    else:
        torch.save(state, os.path.join(folder, "ckpt_best.pth.tar"))

