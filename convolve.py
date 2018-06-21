import torch
import torch.nn.functional as F
from utils import o2t
import numpy as np


def conv(image, kernel, t_tensor=False):
    # TODO: set num in / out chanels

    if not isinstance(kernel, torch.Tensor):
        kernel = np.expand_dims(kernel, axis=0)
        kernel = np.expand_dims(kernel, axis=0)

    image = o2t(image)
    kernel = o2t(kernel)

    res = F.conv2d(image, kernel, padding=int(kernel.size(2) // 2))

    if t_tensor:
        return res

    return res.data
