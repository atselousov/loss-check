import torch
import copy
from utils import o2t


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

        if 'K' in losses:
            raise NotImplementedError('K loss function is not implemented')

        assert len(res_loss) != 0

        return res_loss

    def __call__(self, im1, im2, t_tensor=False):
        # TODO: probably sum of losses must be normalized 
        loss = 0

        im1 = o2t(im1)
        im2 = o2t(im2)

        for l in self._loss:
            loss += l(im1, im2)

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