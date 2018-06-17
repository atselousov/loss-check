import numpy as np
from numba import jit
import pyfftw


@jit('f4(c8[:,:], f4)', cache=True)
def otf2psf(otf, psf_size):
    otf = np.array(otf)

    if np.allclose(otf, 0):
        return np.zeros(psf_size)

    otf_size = np.array(otf.shape)

    otf_placeholder = pyfftw.empty_aligned(otf_size, dtype='complex64')
    otf_placeholder[:, :] = otf

    psf = pyfftw.builders.ifftn(otf_placeholder)()

    num_elements = np.prod(otf_size)

    n_ops = np.sum(np.log2(otf_size) * num_elements)

    if np.max(np.abs(np.imag(psf))) / np.max(np.absolute(psf)) <= n_ops * np.finfo(np.float32).eps:
        psf = np.real(psf)

    for i, dim in enumerate(psf_size):
        psf = np.roll(psf, dim // 2, axis=i)

    i, j = np.meshgrid(range(psf_size[0]), range(psf_size[1]), indexing='ij')

    psf = psf[i, j].reshape(psf_size)

    return psf
