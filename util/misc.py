import numpy as np
from typing import Tuple
import cv2


def convert_range(img: np.ndarray,
                  drange_in: Tuple[int, int],
                  drange_out: Tuple[int, int]
                  ) -> np.ndarray:
    """

    :param img:
    :param drange_in:
    :param drange_out:
    :return:
    """
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        img = img * scale + bias

    return img


def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

def load_image(image_path: str, to_rgb: bool = True, size=None) -> np.ndarray:
    """
    Loads image using OpenCV and cast it from BGR to RGB if necessary
    :param image_path:
    :param to_rgb:
    :return:
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('Image {0} wasn\'t loaded!'.format(image_path))

    if to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if size is not None:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return image


def save_image(path: str, img: np.ndarray, to_bgr: bool = True) -> None:

    if to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)
