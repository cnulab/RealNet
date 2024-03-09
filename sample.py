import warnings
import argparse
import torch
from easydict import EasyDict
import yaml
import os
import numpy as np
from utils.dist_helper import setup_distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from samples.spaced_sample import SpacedDiffusionBeatGans
from models.sdas.create_models import create_diffusion_unet,create_classifier_unet
import math
import random
from utils.categories import CategoriesDict


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description="SDAS sampling")
parser.add_argument("--config", default="./experiments/{}/sample.yaml")
parser.add_argument("--dataset", default="MVTec-AD",choices=['MVTec-AD','VisA','MPDD','BTAD'])
parser.add_argument("--local_rank", default=-1, type=int)


def update_config(config,world_size):

    assert config.dataset.H ==config.dataset.W
    config.unet.use_fp16 = False

    config.classifier.image_size = config.dataset.H
    config.unet.image_size = config.dataset.H

    config.iter_number = int(math.ceil(config.sample_number/(world_size*config.dataset.batch_size)))

    return config


def random_between_a_and_b(a , b):
    assert b>=a
    return random.random()*(b-a)+a



def main():
    args = parser.parse_args()

    args.class_name_dict = CategoriesDict[args.dataset]
    args.config=args.config.format(args.dataset)

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    rank, world_size = setup_distributed()

    config = update_config(config, world_size)

    test_sampler = SpacedDiffusionBeatGans(**config.TestSampler)

    model = create_diffusion_unet(**config.unet).cuda()
    model.eval()

    classifier = create_classifier_unet(**config.classifier).cuda()
    classifier.eval()

    local_rank = int(os.environ["LOCAL_RANK"])

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    classifier = DDP(
        classifier,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    def cond_fn(x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            guided_grad = torch.autograd.grad(selected.sum(), x_in)[0]
            return guided_grad * config.classifier_scale

    device=torch.device('cuda')
    B, C, H, W = config.dataset.batch_size , config.dataset.channels , config.dataset.H , config.dataset.W

    for class_name in args.class_name_dict:

        class_idx=args.class_name_dict[class_name]

        config.export_path = os.path.join(config.workspace.root, class_name)

        os.makedirs(config.export_path, exist_ok=True)

        y = torch.from_numpy(np.array([class_idx])).repeat(B).to(device)

        if rank==0:
            iterator = iter(tqdm(range(config.iter_number),desc='sample {}'.format(class_name)))
        else:
            iterator = iter(range(config.iter_number))

        with torch.no_grad():

            for i in iterator:

                xt = torch.randn((B,C,H,W)).to(device)

                s = random_between_a_and_b(a=0.1,b=0.2)
                # s=0  # generate normal samples

                x1 = test_sampler.p_sample_loop(
                    model=model,
                    noise=xt,
                    device=device,
                    cond_fn=cond_fn, # If you do not use the guided classifier, please comment out this line.
                    model_kwargs={'y': y},
                    s=s,
                )

                # for ddim, set s \in [0.01,0.03]
                x1 = np.clip((x1.cpu().numpy().transpose(0 ,2, 3, 1) + 1) * 127.5, a_min=0, a_max=255).astype(np.uint8)

                for idx in range(B):
                    Image.fromarray(x1[idx]).save(os.path.join(config.export_path,"rank{}_{}.jpg".format(rank,i*B+idx)))


if __name__ == "__main__":
    main()