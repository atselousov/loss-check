import torch
import numpy as np


def o2t(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    elif isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    elif isinstance(obj, list):
        return torch.FloatTensor(obj)
    else:
        raise AttributeError('Not supported type: {}'.format(type(obj)))


def i2t(image):
    image = np.expand_dims(image, axis=0)

    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    else:
        image = np.transpose(image, axes=(0, 3, 1, 2))
    
    return image


def t2i(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().numpy()

    tensor = np.squeeze(tensor)
    
    if len(tensor.shape) == 3:
       tensor = np.transpose(tensor, axes=(1, 2, 0))
    
    return tensor 
