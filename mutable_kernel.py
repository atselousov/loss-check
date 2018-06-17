class MutableKernel(object):
    def __init__(self, init_kernel):
        self._init_kernel = init_kernel[:]

        self._possible_number = 1
        self._i = 0

    def _generate_kernel(self):
        ''' Return generated kernel '''
        # TODO: not correct implementation
        return self._init_kernel

    def __len__(self):
        ''' Return possible number of generated kernels '''
        return self._possible_number

    def __iter__(self):
        return self

    def __next__(self):
        if self._i < self._possible_number:
            self._i += 1
            return self._generate_kernel()
        else:
            raise StopIteration()


