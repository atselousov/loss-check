import copy

import torch
import torch.nn as nn

import torchvision
from utils import o2t


class PerceptualLossVgg11(nn.Module):
    def __init__(self):
        super(PerceptualLossVgg11, self).__init__()

        self.vgg11_conv = torchvision.models.vgg11(pretrained=True).features[:-2]
        self.peprocess = torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])

        for param in self.vgg11_conv:
            param.requires_grad = False

    def forward(self, original, distorted):
        ''' input images must have values in [0, 1] '''
        if original.size(1) == 1 and distorted.size(1) == 1:
            original = torch.cat(3 * [original], dim=1)
            distorted = torch.cat(3 * [distorted], dim=1)

        return torch.mean(torch.abs(self.vgg11_conv(original) - self.vgg11_conv(distorted)))


class Loss(object):
    def __init__(self, losses):
        self._loss = self.__build_loss(losses)
        self._str_losses = copy.deepcopy(losses)

    def __build_loss(self, losses):
        res_loss = []
        if 'L1' in losses:
            res_loss.append(torch.nn.L1Loss())

        if 'L2' in losses:
            res_loss.append(torch.nn.MSELoss())

        if 'P' in losses:
            res_loss.append(PerceptualLossVgg11())

        assert len(res_loss) != 0

        return res_loss

    def __call__(self, im1, im2, t_tensor=False):
        loss = 0

        im1 = o2t(im1)
        im2 = o2t(im2)

        for l in self._loss:
            loss += l(im1, im2)

        loss /= len(self._loss)

        if t_tensor:
            return loss

        return loss.item()

    def __str__(self):
        return '+'.join(self._str_losses)


if __name__ == '__main__':
    import numpy as np
    test_loss = Loss(['L1', 'L2'])

    print(test_loss(np.ones((10, 10)), np.zeros((10, 10))))
    print(test_loss(np.ones((10, 10)), np.zeros((10, 10)), t_tensor=True))
