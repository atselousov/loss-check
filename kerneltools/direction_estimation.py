import numpy as np
from numba import jit
import pyfftw
from tools.psf2otf import psf2otf
from .noise_estimation import get_sigma
import math


#@jit('f4(f4[:,:])', cache=True)
def get_direction(image):
    image_size = image.shape

    dx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    dy = np.flipud(dx.T)

    dx_fft = psf2otf(dx, image_size)
    dy_fft = psf2otf(dy, image_size)

    float_placeholder = pyfftw.empty_aligned(image_size, dtype='float32')
    complex_placeholder = pyfftw.empty_aligned(image_size, dtype='complex64')

    float_placeholder[:, :] = image
    image_fft = pyfftw.builders.fftn(float_placeholder)()

    derivative_dx = dx_fft * image_fft
    derivative_dy = dy_fft * image_fft

    complex_placeholder[:, :] = derivative_dx
    derivative_dx = np.real(pyfftw.builders.ifftn(complex_placeholder)())

    complex_placeholder[:, :] = derivative_dy
    derivative_dy = np.real(pyfftw.builders.ifftn(complex_placeholder)())

    derivative_dx = derivative_dx[5:-5, 5:-5]
    derivative_dy = derivative_dy[5:-5, 5:-5]

    noise_sigma = get_sigma(image)

    cov_matrix = np.cov(derivative_dx.flatten(), derivative_dy.flatten())

    cov_matrix[0, 0] -= 2 * noise_sigma ** 2
    cov_matrix[1, 1] -= 2 * noise_sigma ** 2

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    min_eigenvalues_i = np.argmin(eigenvalues)

    angle = math.atan(eigenvectors[1, min_eigenvalues_i] / eigenvectors[0, min_eigenvalues_i]) * 180. / math.pi

    while angle < 0:
        angle += 180

    return angle
