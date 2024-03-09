import warnings
import argparse
import torch
from easydict import EasyDict
import yaml
import os
import logging
import pprint
from utils.misc_helper import set_seed,get_current_time,create_logger,AverageMeter
from datasets.data_builder import build_dataloader
from samples.tsamples import UniformSampler
from samples.spaced_sample import SpacedDiffusionBeatGans
from models.sdas.create_models import create_classifier_unet
from utils.optimizer_helper import get_optimizer
from utils.criterion_helper import build_criterion
from utils.misc_helper import save_checkpoint
from utils.dist_helper import setup_distributed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.categories import  Categories


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description="train diffusion model guided classifier")
parser.add_argument("--config", default="experiments/{}/classifier.yaml")
parser.add_argument("--dataset", default="MVTec-AD",choices=['MVTec-AD','VisA','MPDD','BTAD'])
parser.add_argument("--local_rank", default=-1, type=int)


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = torch.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def update_config(config,args):
    config.dataset.class_name_list = args.class_name_list
    config.classifier.image_size = config.dataset.input_size[0]
    return config


def main():
    args = parser.parse_args()

    args.class_name_list = Categories[args.dataset]
    args.config=args.config.format(args.dataset)

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    rank, world_size = setup_distributed()

    set_seed(config.random_seed)

    config=update_config(config,args)

    config.exp_path = os.path.dirname(args.config)
    config.checkpoints_path = os.path.join(config.exp_path, config.saver.checkpoints_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)

    train_loader, _ = build_dataloader(config.dataset, distributed=True)

    if rank==0:
        os.makedirs(config.checkpoints_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)

        current_time = get_current_time()

        logger = create_logger(
            "sdas_classifier_logger", config.log_path + "/sdas_classifier_{}.log".format(current_time)
        )

        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))
        logger.info("train_loader len is {}".format(len(train_loader)))

    local_rank = int(os.environ["LOCAL_RANK"])

    train_sampler = SpacedDiffusionBeatGans(**config.TrainSampler)
    Tsampler = UniformSampler(train_sampler)

    model = create_classifier_unet(**config.classifier).cuda()

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    optimizer=get_optimizer(model.parameters(), config.trainer.optimizer)

    last_epoch = 0

    for epoch in range(last_epoch, config.trainer.max_epoch):

        last_iter = epoch * len(train_loader)

        train_loader.sampler.set_epoch(epoch)

        top_1_acc = train_one_epoch(
            config,
            train_loader,
            model,
            optimizer,
            Tsampler,
            train_sampler,
            epoch,
            last_iter,
        )

        if rank==0 and (epoch + 1) % config.trainer.save_freq_epoch == 0:
            logger.info(" * Top 1 acc {:.5f}".format(top_1_acc))
            save_checkpoint(
                                {
                                        "epoch": epoch + 1,
                                        "arch": config,
                                        "state_dict": model.state_dict(),
                                        "acc": top_1_acc,
                                },
                                    config,
                                    epoch=epoch+1
                            )


def train_one_epoch(
    config,
    train_loader,
    model,
    optimizer,
    Tsampler,
    sampler,
    epoch,
    start_iter,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        logger = logging.getLogger("sdas_classifier_logger")

    losses = AverageMeter(config.trainer.print_freq_step)
    criterion = build_criterion(config.criterion)
    model.train()

    epoch_preds = []
    epoch_labels = []

    for i, input in enumerate(train_loader):

        curr_step = start_iter + i

        imgs , class_labels = input['image'].cuda(), input['class_id'].cuda()

        x_start = imgs
        t, weight = Tsampler.sample(len(x_start), x_start.device)
        x_input = sampler.q_sample(x_start,t)

        pred = model(x_input,timesteps=t)

        loss = []
        for name, criterion_loss in criterion.items():
            weight = criterion_loss.weight
            loss.append(weight * criterion_loss({'pred':pred,'label':class_labels}))

        loss = torch.sum(torch.stack(loss))

        reduced_loss = loss.clone()
        dist.all_reduce(reduced_loss)
        reduced_loss = reduced_loss / world_size

        losses.update(reduced_loss.item())

        epoch_preds.append(pred)
        epoch_labels.append(class_labels)

        optimizer.zero_grad()
        loss.backward()

        if config.trainer.get("clip_max_norm", None):
            max_norm = config.trainer.clip_max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        if rank==0 and (curr_step % config.trainer.print_freq_step==0):
            logger.info(
                    "Epoch: [{0}/{1}]\t"
                    "Iter: [{2}/{3}]\t"
                    "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                    .format(
                        epoch + 1,
                        config.trainer.max_epoch,
                        curr_step + 1,
                        len(train_loader) * config.trainer.max_epoch,
                        loss=losses,
                    )
            )


    epoch_preds = torch.cat(epoch_preds,dim=0)
    epoch_labels = torch.cat(epoch_labels,dim=0)

    all_preds=[epoch_preds for _ in range(world_size)]
    all_labels=[epoch_labels for _ in range(world_size)]

    dist.all_gather(all_preds,epoch_preds)
    dist.all_gather(all_labels,epoch_labels)

    all_labels = torch.cat(all_labels,dim=0)
    all_preds = torch.cat(all_preds,dim=0)

    return compute_top_k(all_preds,all_labels,1)


if __name__ == "__main__":
    main()