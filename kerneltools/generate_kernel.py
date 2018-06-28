import warnings
from functools import wraps
import numpy as np
import skimage.transform as transform


def kernel_threshold(kernel, threshold=0.7):
    kernel = kernel.astype(np.float32)
    kernel[kernel < np.max(kernel) * threshold] = 0
    kernel[kernel > 0] = 1
    kernel /= np.sum(kernel)

    return kernel


def catch_warning(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('error', np.RankWarning)
            try:
                return fun(*args, **kwargs)
            except np.RankWarning:
                return None

    return wrapper


@catch_warning
def generate_kernel(x, y, result_size, order=2):
    assert len(x) == len(y) and len(x) == order + 1
    assert np.all(x >= 0) and np.all(y >= 0)

    x_ = 1000 * x
    y_ = 10 * y

    fun = np.poly1d(np.polyfit(x_, y_, order))         

    x_ = np.arange(np.min(x_), np.max(x_), 0.01)
    
    # TODO: catch poorly conditioned examples
    y_ = fun(x_).astype(np.int32)
    y_[y_ < 0] = 0
    x_ = x_.astype(np.int32)
    x_[x_ < 0] = 0

    kernel = np.zeros((np.max(y_) + 1, np.max(x_) + 1))
    kernel[y_, x_] = 1

    current_size = 100

    while current_size > result_size:
        kernel = kernel_threshold(transform.resize(kernel, 
                                                   (current_size, current_size), 
                                                   5, mode='constant'), 0.3)

        current_size *= 0.75
        current_size = int(current_size)

        if current_size < result_size:
            current_size = result_size
            kernel = kernel_threshold(transform.resize(kernel, 
                                                       (current_size, current_size), 
                                                       5, mode='constant'), 0.5)

    return kernel


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    kernel = generate_kernel(np.array([1, 5, 11]), np.array([8, 4, 9]), 11)
    # kernel = generate_kernel(np.array([1, 5, 11]), np.array([1, 5, 11]), 11)
    
    plt.figure()
    plt.imshow(kernel, 'gray')
    plt.show()
