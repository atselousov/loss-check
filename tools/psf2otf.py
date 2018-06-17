import numpy as np
from numba import jit
import pyfftw


@jit('f4[:,:](f4[:,:], f4)', cache=True)
def psf2otf(psf, image_size):
    """Return not shifted fft of psf with size equals to image size"""
    if np.allclose(psf, 0):
        return np.zeros(image_size)

    psf_size = psf.shape

    padded_psf = np.zeros(image_size)
    padded_psf[:psf.shape[0], :psf.shape[1]] = psf

    for i, size in enumerate(psf_size):
        padded_psf = np.roll(padded_psf, - (size // 2), axis=i)

    padded_psf_placeholder = pyfftw.empty_aligned(image_size, dtype='float32')
    padded_psf_placeholder[:, :] = padded_psf
    otf = pyfftw.builders.fftn(padded_psf_placeholder)()

    psf_size = np.array(psf_size)
    n_elem = psf_size.prod()
    n_ops = np.sum(psf_size * np.log2(psf_size) * (n_elem / psf_size))

    threshold = np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf))

    if threshold <= n_ops * np.finfo(float).eps:
        otf = np.real(otf)

    return otf
