from __future__ import division
import copy
import json
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from datasets.base_dataset import BaseDataset, TestBaseTransform, TrainBaseTransform
from datasets.image_reader import build_image_reader
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler


def build_sdas_dataloader(cfg, training, distributed):
    image_reader = build_image_reader(cfg.image_reader)

    if training:
        transform_fn = TrainBaseTransform(
            cfg["input_size"], cfg["hflip"], cfg["vflip"], cfg["rotate"]
        )
    else:
        transform_fn = TestBaseTransform(cfg["input_size"])


    dataset = SDASDataset(
        image_reader,
        cfg["meta_file"],
        training,
        cfg.class_name_list,
        resize=cfg['input_size'],
        transform_fn=transform_fn,
        normalize_fn=None,
    )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
    )

    return data_loader


class SDASDataset(BaseDataset):
    def __init__(
        self,
        image_reader,
        meta_file,
        training,
        class_name_list,
        resize,
        transform_fn,
        normalize_fn,
    ):

        self.resize=resize
        self.image_reader = image_reader
        self.meta_file = meta_file
        self.training = training
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.class_name_dict={ name:i for i,name in enumerate(class_name_list)}

        # construct metas
        with open(meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)


    def __len__(self):
        return len(self.metas)


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
                "label": label,
            }
        )

        if meta.get("clsname", None):
            input["clsname"] = meta["clsname"]
        else:
            input["clsname"] = filename.split("/")[-4]

        input['class_id']= self.class_name_dict[input['clsname']]

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

        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        if self.normalize_fn:
            image = self.normalize_fn(image)
        else:
            image = image * 2.0 - 1

        input.update({"image": image, "mask": mask})
        return input