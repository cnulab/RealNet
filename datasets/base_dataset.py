from __future__ import division

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import datasets.transforms as T


class BaseDataset(Dataset):
    """
    A dataset should implement
        1. __len__ to get size of the dataset, Required
        2. __getitem__ to get a single data, Required

    """
    def __init__(self):
        super(BaseDataset, self).__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TrainBaseTransform(object):
    """
    Resize, flip, rotation for image and mask
    """
    def __init__(self, input_size, hflip, vflip, rotate):
        self.input_size = input_size  # h x w
        self.hflip = hflip
        self.vflip = vflip
        self.rotate = rotate

    def __call__(self, image, mask):
        transform_fn = transforms.Resize(self.input_size, Image.BILINEAR)
        image = transform_fn(image)
        transform_fn = transforms.Resize(self.input_size, Image.NEAREST)
        mask = transform_fn(mask)
        if self.hflip:
            transform_fn = T.RandomHFlip()
            image, mask = transform_fn(image, mask)
        if self.vflip:
            transform_fn = T.RandomVFlip()
            image, mask = transform_fn(image, mask)
        if self.rotate:
            transform_fn = T.RandomRotation([0, 90, 180, 270])
            image, mask = transform_fn(image, mask)
        return image, mask


class TestBaseTransform(object):
    """
    Resize for image and mask
    """
    def __init__(self, input_size):
        self.input_size = input_size  # h x w

    def __call__(self, image, mask):
        transform_fn = transforms.Resize(self.input_size, Image.BILINEAR)
        image = transform_fn(image)
        transform_fn = transforms.Resize(self.input_size, Image.NEAREST)
        mask = transform_fn(mask)
        return image, mask
