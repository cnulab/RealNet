import warnings
import argparse
import torch
from easydict import EasyDict
import yaml
import os
import logging
import numpy as np
import pprint
from torch.cuda.amp import autocast
from utils.misc_helper import set_seed,get_current_time,create_logger,AverageMeter
from datasets.data_builder import build_dataloader
import copy
from samples.tsamples import UniformSampler
from samples.spaced_sample import SpacedDiffusionBeatGans
from models.sdas.create_models import create_diffusion_unet
from utils.misc_helper import ema
from utils.optimizer_helper import get_optimizer
from utils.criterion_helper import build_criterion
from utils.misc_helper import save_checkpoint
from utils.visualize import export_sdas_images
from utils.dist_helper import setup_distributed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from utils.categories import Categories


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description="train diffusion models")
parser.add_argument("--config", default="experiments/{}/diffusion.yaml")
parser.add_argument("--dataset", default="MVTec-AD",choices=['MVTec-AD','VisA','MPDD','BTAD'])
parser.add_argument("--local_rank", default=-1, type=int)


@torch.no_grad()
def SDAS_sample(imgs,class_labels,model,sampler):

    device=torch.device('cuda')

    xt = torch.randn_like(imgs).to(device)

    x0_normal = sampler.p_sample_loop(model=model,
                            noise=xt,
                            device=device,
                            s=0.0,
                            model_kwargs={'y':class_labels})

    x0_week = sampler.p_sample_loop(model=model,
                            noise=xt,
                            device=device,
                            s=0.1,
                            model_kwargs={'y':class_labels})

    x0_strong = sampler.p_sample_loop(model=model,
                            noise=xt,
                            device=device,
                            s=0.2,
                            model_kwargs={'y':class_labels})

    x_gen=torch.cat([x0_normal,x0_week,x0_strong],dim=3)

    xt_det = sampler.ddim_reverse_sample_loop(
        model=model,
        x=imgs,
        clip_denoised=True,
        device=device,
        model_kwargs={'y': class_labels})['sample']


    x0_det = sampler.ddim_sample_loop(model=model,
                            noise=xt_det,
                            eta=0.0,
                            device=device,
                            model_kwargs={'y':class_labels})


    x_recon=torch.cat([imgs,xt_det,x0_det],dim=3)

    return x_recon,x_gen,x0_det


def update_config(config,args):
    config.dataset.class_name_list = args.class_name_list
    config.unet.image_size = config.dataset.input_size[0]
    config.unet.use_fp16 = config.trainer.use_fp16
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
    config.vis_path = os.path.join(config.exp_path, config.saver.vis_dir)

    train_loader, val_loader = build_dataloader(config.dataset, distributed=True)

    if rank==0:
        os.makedirs(config.checkpoints_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)
        os.makedirs(config.vis_path, exist_ok=True)

        current_time = get_current_time()

        logger = create_logger(
            "sdas_diffusion_logger", config.log_path + "/sdas_diffusion_{}.log".format(current_time)
        )
        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))
        logger.info("train_loader len is {}".format(len(train_loader)))

    local_rank = int(os.environ["LOCAL_RANK"])

    train_sampler = SpacedDiffusionBeatGans(**config.TrainSampler)
    test_sampler = SpacedDiffusionBeatGans(**config.TestSampler)
    Tsampler = UniformSampler(train_sampler)

    model = create_diffusion_unet(**config.unet).cuda()

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)
    ema_model.eval()

    optimizer = get_optimizer(model.parameters(), config.trainer.optimizer)

    last_epoch = 0

    for epoch in range(last_epoch, config.trainer.max_epoch):

        last_iter = epoch * len(train_loader)

        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        train_one_epoch(
            config,
            train_loader,
            model,
            ema_model,
            optimizer,
            Tsampler,
            train_sampler,
            epoch,
            last_iter,
        )

        if (epoch + 1) % config.trainer.val_freq_epoch == 0:
            fileinfos, gen_images, recon_images, avg_loss = validate(config, val_loader, ema_model, test_sampler, epoch + 1)
            export_sdas_images(config, fileinfos, gen_images, recon_images, epoch+1)
            if rank == 0:
                save_checkpoint(
                                {
                                        "epoch": epoch + 1,
                                        "arch": config,
                                        "state_dict": ema_model.state_dict(),
                                        "loss": avg_loss,
                                },
                                    config,
                                    epoch=epoch+1
                            )


def train_one_epoch(
    config,
    train_loader,
    model,
    ema_model,
    optimizer,
    Tsampler,
    sampler,
    epoch,
    start_iter,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        logger = logging.getLogger("sdas_diffusion_logger")

    losses = AverageMeter(config.trainer.print_freq_step)
    model.train()

    for i, input in enumerate(train_loader):

        curr_step = start_iter + i

        imgs , class_labels = input['image'].cuda(), input['class_id'].cuda()

        x_start = imgs
        t, weight = Tsampler.sample(len(x_start), x_start.device)

        with autocast(enabled=config.trainer.use_fp16):

            terms = sampler.training_losses(model=model,x_start=x_start, t=t, model_kwargs={'y':class_labels})
            loss = terms['loss'].mean()

        reduced_loss = loss.clone()
        dist.all_reduce(reduced_loss)
        reduced_loss = reduced_loss / world_size
        losses.update(reduced_loss.item())

        step_context = model.no_sync if curr_step % config.trainer.accumulate != 0 else nullcontext
        with step_context():
            loss.backward()
            if config.trainer.get("clip_max_norm", None):
                max_norm = config.trainer.clip_max_norm
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if step_context == nullcontext:
                optimizer.step()
                optimizer.zero_grad()
                ema(model, ema_model, config.trainer.ema_decay)

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



def validate(config, val_loader, model, test_sample, epoch):

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model.eval()

    losses = []
    criterion = build_criterion(config.criterion)

    x_recon_images = []
    x_gen_images = []
    fileinfos = []

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            # forward
            imgs , class_labels = input['image'].cuda(), input['class_id'].cuda()

            x_recon, x_gen, x0_det = SDAS_sample(imgs, class_labels, model, test_sample)

            for j in range(len(input['filename'])):
                fileinfos.append(
                    {
                        "filename": str(input["filename"][j]),
                        "clsname": str(input["clsname"][j]),
                    }
                )

            x_gen_images.append(x_gen)
            x_recon_images.append(x_recon)

            l1 = []
            for name, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                l1.append(weight * criterion_loss({"ori": imgs, "recon": x0_det}))

            l1 = torch.sum(torch.stack(l1))

            dist.all_reduce(l1)
            l1 = l1 / world_size
            losses.append(l1.item())

            if i == config.trainer.val_batch_number:
                break

    avg_loss = np.mean(losses)

    if rank==0:
        logger = logging.getLogger("sdas_diffusion_logger")
        logger.info(" * Loss_sum {:.5f}".format(avg_loss))

    gen_images = torch.cat(x_gen_images, dim=0).cpu().detach().numpy()
    recon_images = torch.cat(x_recon_images, dim=0).cpu().detach().numpy()

    return fileinfos, gen_images, recon_images, avg_loss


if __name__ == "__main__":
    main()