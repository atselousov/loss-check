import numpy as np
from numba import jit
import pyfftw


#@jit('f4(f4[:,:], f4)', cache=True)
def get_sigma(image, radius_coefficient=0.1):

    image_size = image.shape
    h_center, w_center = image_size[0] // 2, image_size[1] // 2

    image_placeholder = pyfftw.empty_aligned(image_size, dtype='float32')
    image_placeholder[:, :] = image

    image_spectrum = pyfftw.builders.fft2(image_placeholder)()
    image_spectrum = np.abs(image_spectrum)

    h_indexes, w_indexes = np.meshgrid(range(image_size[0]), range(image_size[1]), indexing='ij')
    mask = np.sqrt((h_indexes - h_center) ** 2 + (w_indexes - w_center) ** 2)

    radius = radius_coefficient * (np.min(image_size) // 2)
    mask[mask >= radius] = 0.
    mask[mask > 0] = 1.

    sectors = [np.zeros(image_size), np.zeros(image_size), np.ones(image_size)]

    sectors[0][:h_center, :w_center] = 1.
    sectors[0][h_center:, w_center:] = 1.

    sectors[1][:h_center, w_center:] = 1.
    sectors[1][h_center:, :w_center] = 1.

    sigmas = []

    for sector in sectors:
        sub_mask = sector * mask
        num_nonzero_elements = np.sum(sub_mask)
        sigmas.append(np.sqrt(np.sum((sub_mask * image_spectrum) ** 2) / (num_nonzero_elements * np.prod(image_size))))

    return np.min(sigmas)
