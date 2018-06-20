import numpy as np
from itertools import product
from kerneltools.generate_kernel import generate_kernel


class MutableKernel(object):
    def __init__(self, init_kernel):
        self._init_kernel = init_kernel[:]
        self._kernel_size = self._init_kernel.shape[0]

        self._possible_number = np.product(self._init_kernel.shape)

        y, x = np.where(self._init_kernel > 0)

        self.x = np.array([x[0], 0, x[-1]])
        self.y = np.array([y[0], 0, y[-1]])

        self._ind_gen = product(range(self._kernel_size), repeat=2)

    def __len__(self):
        ''' Return possible number of generated kernels '''
        return self._possible_number

    def __iter__(self):
        return self

    def __next__(self):
        y, x = next(self._ind_gen)

        self.x[1] = x
        self.y[1] = y
        
        return x, y, generate_kernel(self.x, self.y, self._kernel_size)
