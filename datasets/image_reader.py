import os
import cv2


class OpenCVReader:
    def __init__(self, image_dir, color_mode):
        self.image_dir = image_dir
        self.color_mode = color_mode
        assert color_mode in ["RGB", "BGR", "GRAY"], f"{color_mode} not supported"
        if color_mode != "BGR":
            self.cvt_color = getattr(cv2, f"COLOR_BGR2{color_mode}")
        else:
            self.cvt_color = None

    def __call__(self, filename, is_mask=False):
        filename = os.path.join(self.image_dir, filename)
        assert os.path.exists(filename), filename
        if is_mask:
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            return img

        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        if self.color_mode != "BGR":
            img = cv2.cvtColor(img, self.cvt_color)
        return img



def build_image_reader(cfg_reader):
    if cfg_reader["type"] == "opencv":
        return OpenCVReader(**cfg_reader["kwargs"])

    else:
        raise TypeError("no supported image reader type: {}".format(cfg_reader["type"]))
