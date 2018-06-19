import numpy as np
import pyfftw
from .direction_estimation import get_direction
from numba import jit
from tools.gauss_2d import gauss_2d


#@jit('UniTuple(i4, 2)(f4[:,:])', cache=True)
def get_length(image):
    distortion_angle = get_direction(image)

    image_size = image.shape
    h_center, w_center = image_size[0] // 2, image_size[1] // 2

    float_placeholder = pyfftw.empty_aligned(image_size, dtype='float32')
    complex_placeholder = pyfftw.empty_aligned(image_size, dtype='complex64')

    float_placeholder[:, :] = image
    image_cepstrum = pyfftw.builders.fftn(float_placeholder)()
    image_cepstrum = np.log(np.absolute(image_cepstrum))

    complex_placeholder[:, :] = image_cepstrum
    image_cepstrum = np.absolute(pyfftw.builders.ifftn(complex_placeholder)())

    image_cepstrum = np.roll(image_cepstrum, h_center, axis=0)
    image_cepstrum = np.roll(image_cepstrum, w_center, axis=1)

    smooth_mask = gauss_2d(image_size, sigma=10)
    smooth_mask = np.max(smooth_mask) - smooth_mask

    sector_mask = np.zeros(image_size)

    if 0 <= distortion_angle < 90 or 180 <= distortion_angle < 270:
        sector_mask[:h_center, w_center:] = 1
        sector_mask[h_center:, :w_center] = 1
    else:
        sector_mask[:h_center, :w_center:] = 1
        sector_mask[h_center:, w_center:] = 1

    masked_image_cepstrum = image_cepstrum * smooth_mask * sector_mask

    max_index = np.argmax(masked_image_cepstrum)

    h_max_indexes, w_max_indexes = np.abs(max_index // image_size[1] - h_center), \
                                   np.abs(max_index % image_size[1] - w_center)

    lenght = np.sqrt(h_max_indexes ** 2 + w_max_indexes ** 2)

    return lenght, distortion_angle

