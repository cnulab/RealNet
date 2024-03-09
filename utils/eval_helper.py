import glob
import logging
import os
import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from sklearn import metrics
from skimage import measure


class Report:
    def __init__(self, heads=None):
        if heads:
            self.heads = list(map(str, heads))
        else:
            self.heads = ()
        self.records = []

    def add_one_record(self, record):
        if self.heads:
            if len(record) != len(self.heads):
                raise ValueError(
                    f"Record's length ({len(record)}) should be equal to head's length ({len(self.heads)})."
                )
        self.records.append(record)

    def __str__(self):
        return tabulate.tabulate(
            self.records,
            self.heads,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )



class EvalDataMeta:
    def __init__(self, preds, masks):
        self.preds = preds  # N x H x W
        self.masks = masks  # N x H x W


class EvalImage:
    def __init__(self, data_meta, **kwargs):
        self.preds = self.encode_pred(data_meta.preds, **kwargs)
        self.masks = self.encode_mask(data_meta.masks)
        self.preds_good = sorted(self.preds[self.masks == 0], reverse=True)
        self.preds_defe = sorted(self.preds[self.masks == 1], reverse=True)
        self.num_good = len(self.preds_good)
        self.num_defe = len(self.preds_defe)

    @staticmethod
    def encode_pred(preds):
        raise NotImplementedError

    def encode_mask(self, masks):
        N, _, _ = masks.shape
        masks = (masks.reshape(N, -1).sum(axis=1) != 0).astype(np.int)  # (N, )
        return masks

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc



class EvalImageMax(EvalImage):
    @staticmethod
    def encode_pred(preds, avgpool_size):
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        preds = (
            F.avg_pool2d(preds, avgpool_size, stride=1).cpu().numpy()
        )  # N x 1 x H x W
        return preds.reshape(N, -1).max(axis=1)  # (N, )



class EvalPixelAUC:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc


class EvalPixelPro:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [np.expand_dims(pred,axis=0) for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [np.expand_dims(mask,axis=0)  for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1
        self.masks=self.masks.astype(np.bool)

    def rescale(self,x):
        return (x - x.min()) / (x.max() - x.min())

    def eval_auc(self):

        max_step = 1000
        expect_fpr = 0.3  # default 30%

        max_th = self.preds.max()
        min_th = self.preds.min()
        delta = (max_th - min_th) / max_step

        pros_mean = []
        pros_std = []

        threds = []
        fprs = []

        binary_score_maps = np.zeros_like(self.preds, dtype=np.bool)

        for step in range(max_step):
            thred = max_th - step * delta
            # segmentation
            binary_score_maps[self.preds <= thred] = 0
            binary_score_maps[self.preds > thred] = 1
            pro = []  # per region overlap
            # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
            # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
            for i in range(len(binary_score_maps)):  # for i th image
                # pro (per region level)
                label_map = measure.label(self.masks[i], connectivity=2)
                props = measure.regionprops(label_map)
                for prop in props:
                    x_min, y_min, x_max, y_max = prop.bbox  # find the bounding box of an anomaly region
                    cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                    # cropped_mask = gt_mask[i][x_min:x_max, y_min:y_max]   # bug!
                    cropped_mask = prop.filled_image  # corrected!
                    intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                    pro.append(intersection / prop.area)

            pros_mean.append(np.array(pro).mean())
            pros_std.append(np.array(pro).std())
            # fpr for pro-auc
            gt_masks_neg = ~self.masks
            fpr = np.logical_and(gt_masks_neg, binary_score_maps).sum() / gt_masks_neg.sum()
            fprs.append(fpr)
            threds.append(thred)
        # as array
        pros_mean = np.array(pros_mean)
        fprs = np.array(fprs)
        idx = fprs <= expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
        fprs_selected = fprs[idx]
        fprs_selected = self.rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
        pros_mean_selected = pros_mean[idx]
        seg_pro_auc = metrics.auc(fprs_selected, pros_mean_selected)
        return seg_pro_auc


eval_lookup_table = {
    "image": EvalImageMax,
    "pixel": EvalPixelAUC,
    "pro":EvalPixelPro,
}


def performances(class_name, preds, masks, config):
    ret_metrics = {}

    clsnames = [class_name]

    for clsname in clsnames:
        preds_cls = []
        masks_cls = []
        for pred, mask in zip( preds, masks):
            preds_cls.append(pred[None, ...])
            masks_cls.append(mask[None, ...])

        preds_cls = np.concatenate(np.asarray(preds_cls), axis=0)  # N x H x W
        masks_cls = np.concatenate(np.asarray(masks_cls), axis=0)  # N x H x W

        data_meta = EvalDataMeta(preds_cls, masks_cls)

        if config.get("auc", None):
            for metric in config.auc:
                evalname = metric["name"]
                kwargs = metric.get("kwargs", {})
                eval_method = eval_lookup_table[evalname](data_meta, **kwargs)
                auc = eval_method.eval_auc()
                ret_metrics["{}_{}_auc".format(clsname, evalname)] = auc


    if config.get("auc", None):
        for metric in config.auc:
            evalname = metric["name"]
            evalvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for clsname in clsnames
            ]
            mean_auc = np.mean(np.array(evalvalues))
            ret_metrics["{}_{}_auc".format("mean", evalname)] = mean_auc

    return ret_metrics



def log_metrics(ret_metrics, config,logger_name):
    logger = logging.getLogger(logger_name)
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = list(clsnames - set(["mean","best"]))
    clsnames.sort()
    clsnames = clsnames+ ['best']

    # auc
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

        logger.info(f"\n{record}")
