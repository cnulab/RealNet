import warnings
import argparse
import torch
from datasets.data_builder import build_dataloader
from easydict import EasyDict
import yaml
import os
from utils.misc_helper import set_seed
from models.model_helper import ModelHelper
from utils.eval_helper import performances
from sklearn.metrics import precision_recall_curve
import numpy as np
from utils.visualize import export_segment_images
from utils.eval_helper import Report
from train_realnet import update_config
from utils.categories import Categories


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description="evaluation RealNet")
parser.add_argument("--config", default="experiments/{}/realnet.yaml")
parser.add_argument("--dataset", default="MVTec-AD",choices=['MVTec-AD','VisA','MPDD','BTAD'])
parser.add_argument("--class_name", default="bottle",choices=[
        # mvtec-ad
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
        # visa
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
        #mpdd
        "bracket_black",
        "bracket_brown",
        "bracket_white",
        "connector",
        "metal_plate",
        "tubes",
        # btad
         "01",
         "02",
         "03",
        ] )


def main():
    args = parser.parse_args()

    class_name_list=Categories[args.dataset]

    assert args.class_name in class_name_list

    args.config=args.config.format(args.dataset)

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.exp_path = os.path.dirname(args.config)

    args.checkpoints_folder = os.path.join(config.exp_path, config.saver.checkpoints_dir,args.class_name)

    args.model_path=os.path.join(args.checkpoints_folder,"ckpt_best.pth.tar")

    config=update_config(config,args)
    set_seed(config.random_seed)

    config.evaluator.metrics['auc'].append({'name':'pro'})

    config.vis_path = os.path.join(config.exp_path, config.saver.vis_dir)
    os.makedirs(config.vis_path, exist_ok=True)

    _, val_loader = build_dataloader(config.dataset,distributed=False)

    model = ModelHelper(config.net)
    model.cuda()

    state_dict=torch.load(args.model_path)
    model.load_state_dict(state_dict['state_dict'],strict=False)

    ret_metrics = validate(config,val_loader, model,args.class_name)
    print_metrics(ret_metrics, config.evaluator.metrics, args.class_name)


def print_metrics(ret_metrics, config, class_name):
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = list(clsnames - set(["mean"]))
    clsnames.sort()

    if config.get("auc", None):
        auc_keys = [k for k in ret_metrics.keys() if "auc" in k]
        evalnames = list(set([k.rsplit("_", 2)[1] for k in auc_keys]))
        evalnames.sort()

        record = Report(["clsname"] + evalnames)

        for clsname in clsnames:
            clsvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for evalname in evalnames
            ]
            record.add_one_record([clsname] + clsvalues)

        print(f"\n{record}")



def validate(config,val_loader, model,class_name):

    model.eval()

    fileinfos = []
    preds = []
    masks = []

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            # forward
            outputs = model(input,train=False)

            for j in range(len(outputs['filename'])):
                fileinfos.append(
                    {
                        "filename": str(outputs["filename"][j]),
                        "height": int(outputs["height"][j]),
                        "width": int(outputs["width"][j]),
                        "clsname": str(outputs["clsname"][j]),
                    }
                )
            preds.append(outputs["anomaly_score"].cpu().numpy())
            masks.append(outputs["mask"].cpu().numpy())

    preds = np.squeeze(np.concatenate(np.asarray(preds), axis=0),axis=1)  # N x H x W
    masks = np.squeeze(np.concatenate(np.asarray(masks), axis=0),axis=1)  # N x H x W

    ret_metrics = performances(class_name, preds, masks, config.evaluator.metrics)

    preds_cls = []
    masks_cls = []
    image_paths = []

    for fileinfo, pred, mask in zip(fileinfos, preds, masks):
        preds_cls.append(pred[None, ...])
        masks_cls.append(mask[None, ...])
        image_paths.append(fileinfo['filename'])

    preds_cls = np.concatenate(np.asarray(preds_cls), axis=0)  # N x H x W
    masks_cls = np.concatenate(np.asarray(masks_cls), axis=0)  # N x H x W
    masks_cls[masks_cls != 0.0] = 1.0

    precision, recall, thresholds = precision_recall_curve(masks_cls.flatten(), preds_cls.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    seg_threshold = thresholds[np.argmax(f1)]
    export_segment_images(config, image_paths, masks_cls, preds_cls, seg_threshold, class_name)
    return ret_metrics


if __name__ == "__main__":
    main()
