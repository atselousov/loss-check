import numpy as np
from scipy.ndimage import rotate
from .length_estimation import get_length


def kernel_threshold(kernel, threshold=0.7):
    kernel = kernel.astype(np.float32)
    kernel[kernel < np.max(kernel) * threshold] = 0

    return kernel_norm(kernel)


def kernel_norm(kernel):
    kernel = kernel.astype(np.float32)
    kernel[kernel < 0] = 0
    kernel_sum = np.sum(kernel)

    if kernel_sum == 0:
        raise NameError('All elements of input kernel equal to zero.')

    kernel /= kernel_sum

    return kernel


def initialize_kernel(distorted_image):

    kernel_size, distortion_angle = get_length(distorted_image)

    kernel_size = int(kernel_size)
    # kernel_size = (kernel_size, kernel_size)

    if distortion_angle is None:
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, kernel_size // 2] = 1
    else:
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1
        kernel = rotate(kernel, distortion_angle, reshape=False, mode='nearest')

    return kernel_norm(kernel), kernel_size, distortion_angle