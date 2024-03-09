from __future__ import division
import copy
import json
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from datasets.base_dataset import BaseDataset, TestBaseTransform, TrainBaseTransform
from datasets.image_reader import build_image_reader
from torch.utils.data.sampler import RandomSampler
from glob import glob
import os
import cv2
import torch
import imgaug.augmenters as iaa
import math
from skimage import morphology
from torch.utils.data.distributed import DistributedSampler


def lerp_np(x,y,w):
    fin_out = (y-x)*w + x
    return fin_out

def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients,d[0],axis=0),d[1],axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
    dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])


def build_realnet_dataloader(cfg, training,distributed=True):
    image_reader = build_image_reader(cfg.image_reader)
    normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])

    if training:
        transform_fn = TrainBaseTransform(
            cfg["input_size"], cfg["hflip"], cfg["vflip"], cfg["rotate"]
        )
    else:
        transform_fn = TestBaseTransform(cfg["input_size"])

    dataset = RealNetDataset(
        image_reader,
        cfg["meta_file"],
        training,
        dataset=cfg['type'],
        resize=cfg['input_size'],
        transform_fn=transform_fn,
        normalize_fn=normalize_fn,
        dtd_dir=cfg.get("dtd_dir", None),
        sdas_dir=cfg.get("sdas_dir", None),
        dtd_transparency_range=cfg.get("dtd_transparency_range",[]),
        sdas_transparency_range=cfg.get("sdas_transparency_range", []),
        perlin_scale=cfg.get("perlin_scale",0),
        min_perlin_scale=cfg.get('min_perlin_scale',0),
        anomaly_types=cfg.get('anomaly_types',{}),
    )

    if distributed and training:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
    )
    return data_loader


class RealNetDataset(BaseDataset):
    def __init__(
        self,
        image_reader,
        meta_file,
        training,
        resize,
        transform_fn,
        normalize_fn,
        dataset,
        dtd_dir = None,
        sdas_dir=None,
        dtd_transparency_range = [],
        sdas_transparency_range=[],
        perlin_scale: int = 6,
        min_perlin_scale: int = 0,
        perlin_noise_threshold: float = 0.5,
        anomaly_types={},
    ):

        self.resize=resize
        self.image_reader = image_reader
        self.meta_file = meta_file
        self.training = training
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.anomaly_types = anomaly_types
        self.dataset=dataset

        if training:
            self.dtd_dir=dtd_dir
            self.sdas=sdas_dir

            self.sdas_transparency_range=sdas_transparency_range
            self.dtd_transparency_range=dtd_transparency_range

            self.perlin_scale=perlin_scale
            self.min_perlin_scale=min_perlin_scale
            self.perlin_noise_threshold=perlin_noise_threshold

        # construct metas
        with open(meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)

        if dtd_dir:
            self.dtd_file_list = glob(os.path.join(dtd_dir, '*/*'))

        if sdas_dir:
            self.sdas_file_list = glob(os.path.join(sdas_dir, '*'))

    def __len__(self):
        return len(self.metas)


    def choice_anomaly_type(self):
        if len(self.anomaly_types)!=0 and self.training:
            return np.random.choice(a=[ key for key in self.anomaly_types],
                                    p=[ self.anomaly_types[key]  for key in self.anomaly_types],
                                    size=(1,),replace=False)[0]
        else:
            return 'normal'


    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]
        # read image
        filename = meta["filename"]
        label = meta["label"]

        image = self.image_reader(meta["filename"])

        input.update(
            {
                "filename": filename,
                "height": image.shape[0],
                "width": image.shape[1],
                "label": label,
            }
        )

        if meta.get("clsname", None):
            input["clsname"] = meta["clsname"]
        else:
            input["clsname"] = filename.split("/")[-4]

        image = Image.fromarray(image, "RGB")

        # read / generate mask

        if meta.get("maskname", None):
            mask = self.image_reader(meta["maskname"], is_mask=True)
        else:
            if label == 0:  # good
                mask = np.zeros((image.height, image.width)).astype(np.uint8)
            elif label == 1:  # defective
                mask = (np.ones((image.height, image.width)) * 255).astype(np.uint8)
            else:
                raise ValueError("Labels must be [None, 0, 1]!")

        mask = Image.fromarray(mask, "L")

        if self.transform_fn:
            image, mask = self.transform_fn(image, mask)

        if self.training:
            gt_image = copy.deepcopy(image)
            gt_image = transforms.ToTensor()(gt_image)
            if self.normalize_fn:
                gt_image = self.normalize_fn(gt_image)
            input.update({'gt_image':gt_image})

        image_anomaly_type =self.choice_anomaly_type()
        assert image_anomaly_type in ['normal','dtd','sdas']

        if image_anomaly_type!='normal':
            anomaly_image, anomaly_mask = self.generate_anomaly(np.array(image), self.dataset, input["clsname"],image_anomaly_type)
            image = Image.fromarray(anomaly_image, "RGB")
            mask = Image.fromarray(np.array(anomaly_mask*255.0).astype(np.uint8), "L")

        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        if self.normalize_fn:
            image = self.normalize_fn(image)

        input.update({"image": image, "mask": mask, "anomaly_type":image_anomaly_type})
        return input


    def generate_anomaly(self, img, dataset,subclass, image_anomaly_type ,get_mask_only=False):
        '''
        step 1. generate mask
            - target foreground mask
            - perlin noise mask

        step 2. generate texture or structure anomaly
            - texture: load DTD
            - structure: we first perform random adjustment of mirror symmetry, rotation, brightness, saturation,
            and hue on the input image  𝐼 . Then the preliminary processed image is uniformly divided into a 4×8 grid
            and randomly arranged to obtain the disordered image  𝐼

        step 3. blending image and anomaly source
        '''

        target_foreground_mask = self.generate_target_foreground_mask(img,dataset, subclass)
        # Image.fromarray(target_foreground_mask*255).convert('L').save("foreground.jpg")

        ## perlin noise mask
        perlin_noise_mask = self.generate_perlin_noise_mask()

        ## mask
        mask = perlin_noise_mask * target_foreground_mask

        # step 2. generate texture or structure anomaly
        if get_mask_only:
            return mask

        anomaly_source_img = self.anomaly_source(img=img,
                                                 mask=mask,
                                                 anomaly_type=image_anomaly_type).astype(np.uint8)

        return anomaly_source_img, mask


    def generate_target_foreground_mask(self, img: np.ndarray, dataset:str,subclass: str) -> np.ndarray:

        # convert RGB into GRAY scale
        if dataset=='mvtec':
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if subclass in ['carpet', 'leather', 'tile', 'wood', 'cable', 'transistor']:
                return np.ones_like(img_gray)
            if subclass=='pill':
                _, target_foreground_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                target_foreground_mask = target_foreground_mask.astype(np.bool).astype(np.int)
            elif subclass in ['hazelnut', 'metal_nut', 'toothbrush']:
                _, target_foreground_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
                target_foreground_mask = target_foreground_mask.astype(np.bool).astype(np.int)
            elif subclass in ['bottle','capsule','grid','screw','zipper']:
                _, target_background_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                target_background_mask = target_background_mask.astype(np.bool).astype(np.int)
                target_foreground_mask = 1 - target_background_mask
            else:
                raise NotImplementedError("Unsupported foreground segmentation category")
            target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(6))
            target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(6))
            return target_foreground_mask

        elif dataset=='visa':
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if subclass in ['capsules']:
                return np.ones_like(img_gray)
            if subclass in ['pcb1', 'pcb2', 'pcb3', 'pcb4']:
                _, target_foreground_mask = cv2.threshold(img[:, :, 2], 100, 255,
                                                          cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
                target_foreground_mask = target_foreground_mask.astype(np.bool).astype(np.int)
                target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(8))
                target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(3))
                return target_foreground_mask
            else:
                _, target_foreground_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                target_foreground_mask = target_foreground_mask.astype(np.bool).astype(np.int)
                target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(3))
                target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(3))
                return target_foreground_mask

        elif dataset=='mpdd':
            if subclass in ['bracket_black', 'bracket_brown', 'connector']:
                img_seg = img[:, :, 1]
            elif subclass in ['bracket_white', 'tubes']:
                img_seg = img[:, :, 2]
            else:
                img_seg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            _, target_background_mask = cv2.threshold(img_seg, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            target_background_mask = target_background_mask.astype(np.bool).astype(np.int)

            if subclass in ['bracket_white', 'tubes']:
                target_foreground_mask = target_background_mask
            else:
                target_foreground_mask = 1 - target_background_mask

            target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(6))
            return target_foreground_mask

        elif dataset=='btad':
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            if subclass in ['02']:
                return np.ones_like(img_gray)

            _, target_background_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            target_foreground_mask = target_background_mask.astype(np.bool).astype(np.int)
            target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(15))
            target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(6))
            return target_foreground_mask
        else:
            raise NotImplementedError(f"dataset type '{dataset}' is not supported")


    def generate_perlin_noise_mask(self) -> np.ndarray:
        # define perlin noise scale
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])

        # generate perlin noise
        perlin_noise = rand_perlin_2d_np((self.resize[0], self.resize[1]), (perlin_scalex, perlin_scaley))

        # apply affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)

        # make a mask by applying threshold
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold,
            np.ones_like(perlin_noise),
            np.zeros_like(perlin_noise)
        )
        return mask_noise

    def rand_augment(self):
        augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
        aug = iaa.Sequential([
            augmenters[aug_idx[0]],
            augmenters[aug_idx[1]],
            augmenters[aug_idx[2]]
        ])
        return aug

    def anomaly_source(self, img: np.ndarray,
                             mask:np.ndarray,
                             anomaly_type:str):

        if anomaly_type=='sdas':
            anomaly_source_img=self._sdas_source()
            factor = np.random.uniform(*self.sdas_transparency_range, size=1)[0]

        elif anomaly_type=='dtd':
            anomaly_source_img = self._dtd_source()
            factor = np.random.uniform(*self.dtd_transparency_range, size=1)[0]
        else:
            raise NotImplementedError("unknown ano")

        mask_expanded = np.expand_dims(mask, axis=2)
        anomaly_source_img = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)
        anomaly_source_img = ((- mask_expanded + 1) * img) + anomaly_source_img
        return anomaly_source_img


    def _dtd_source(self) -> np.ndarray:
        idx = np.random.choice(len(self.dtd_file_list))
        dtd_source_img = cv2.imread(self.dtd_file_list[idx])
        dtd_source_img = cv2.cvtColor(dtd_source_img, cv2.COLOR_BGR2RGB)
        dtd_source_img = cv2.resize(dtd_source_img, dsize=self.resize)
        dtd_source_img = self.rand_augment()(image=dtd_source_img)
        return dtd_source_img.astype(np.float32)

    def _sdas_source(self) -> np.ndarray:
        path = random.choice(self.sdas_file_list)
        sdas_source_img = cv2.imread(path)
        sdas_source_img = cv2.cvtColor(sdas_source_img, cv2.COLOR_BGR2RGB)
        sdas_source_img = cv2.resize(sdas_source_img, dsize=self.resize)
        return sdas_source_img.astype(np.float32)
