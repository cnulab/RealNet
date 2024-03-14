from __future__ import division
import numbers
import random
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
)


# Horizontal Flip
class RandomHFlip(object):
    def __init__(self, flip_p=0.5):
        self.flip_p = flip_p

    def __call__(self, img, mask):
        flip_flag = torch.rand(1)[0].item() < self.flip_p
        if flip_flag:
            return F.hflip(img), F.hflip(mask)
        else:
            return img, mask


# Vertical Flip
class RandomVFlip(object):
    def __init__(self, flip_p=0.5):
        self.flip_p = flip_p

    def __call__(self, img, mask):
        flip_flag = torch.rand(1)[0].item() < self.flip_p
        if flip_flag:
            return F.vflip(img), F.vflip(mask)
        else:
            return img, mask


# from POD
class RandomColorJitter(object):
    """
    Randomly change the brightness, contrast and saturation of an image.

    Arguments:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, prob=0):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.prob = prob

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def get_params(self, brightness, contrast, saturation, hue):
        """
        Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        img_transforms = []

        if brightness is not None and random.random() < self.prob:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            img_transforms.append(
                transforms.Lambda(lambda img: adjust_brightness(img, brightness_factor))
            )

        if contrast is not None and random.random() < self.prob:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            img_transforms.append(
                transforms.Lambda(lambda img: adjust_contrast(img, contrast_factor))
            )

        if saturation is not None and random.random() < self.prob:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            img_transforms.append(
                transforms.Lambda(lambda img: adjust_saturation(img, saturation_factor))
            )

        if hue is not None and random.random() < self.prob:
            hue_factor = random.uniform(hue[0], hue[1])
            img_transforms.append(
                transforms.Lambda(lambda img: adjust_hue(img, hue_factor))
            )

        random.shuffle(img_transforms)
        img_transforms = transforms.Compose(img_transforms)

        return img_transforms

    def __call__(self, img):
        """
        Arguments:
            img (PIL Image): Input image.
        Returns:
            img (PIL Image): Color jittered image.
        """
        transform = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        img = transform(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "brightness={0}".format(self.brightness)
        format_string += ", contrast={0}".format(self.contrast)
        format_string += ", saturation={0}".format(self.saturation)
        format_string += ", hue={0})".format(self.hue)
        return format_string

    @classmethod
    def from_params(cls, params):
        brightness = params.get("brightness", 0.1)
        contrast = params.get("contrast", 0.5)
        hue = params.get("hue", 0.07)
        saturation = params.get("saturation", 0.5)
        prob = params.get("prob", 0.5)
        return cls(
            brightness=brightness,
            contrast=contrast,
            hue=hue,
            saturation=saturation,
            prob=prob,
        )


class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            degrees = [degrees]
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.choice(degrees)

        return angle

    def __call__(self, img, mask):
        """
            img, mask (PIL Image): Image to be rotated.
        Returns:
            img, mask (PIL Image): Rotated image.
        """
        angle = self.get_params(self.degrees)
        img = F.rotate(img, angle, self.resample, self.expand, self.center)
        mask = F.rotate(mask, angle, self.resample, self.expand, self.center)
        return img, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + "(degrees={0}".format(self.degrees)
        format_string += ", resample={0}".format(self.resample)
        format_string += ", expand={0}".format(self.expand)
        if self.center is not None:
            format_string += ", center={0}".format(self.center)
        format_string += ")"
        return format_string
