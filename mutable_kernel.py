from itertools import product, combinations

import numpy as np

from kerneltools.generate_kernel import generate_kernel


class MutableKernel(object):
    def __init__(self, kernel_size, linear_kernel=None, order=2):
        self._kernel_size = int(kernel_size)
        self._order = order

        if linear_kernel is not None:
            self._generator = self._init_from_kernel(linear_kernel)
        else:
            self._generator = combinations(product(range(self._kernel_size), repeat=2), self._order + 1)

        self._x = np.zeros((self._order + 1,))
        self._y = np.zeros((self._order + 1,))

    def _init_from_kernel(self, linear_kernel):
        self._order = 2
        assert len(linear_kernel.shape) == 2
        assert self._kernel_size == linear_kernel.shape[0] == linear_kernel.shape[1]

        y, x = np.where(linear_kernel > 0)

        def generator():
            for p in product(range(self._kernel_size), repeat=2):
                yield (x[0], y[0]), p, (x[-1], y[-1])

        return generator()

    def __iter__(self):
        return self

    def __next__(self):
        points = next(self._generator)

        for i, p in enumerate(points):
            self._x[i] = p[0]
            self._y[i] = p[1]
        
        return points, \
               generate_kernel(self._x, self._y, self._kernel_size, order=self._order)
