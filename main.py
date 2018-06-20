import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import *
from convolve import conv
from iom import read_image, write_image
from kerneltools.generate_kernel import generate_kernel
from kerneltools.init_kernel import initialize_kernel, kernel_threshold
from loss import Loss
from mutable_kernel import MutableKernel
from utils import i2t, t2i


def get_args():
    parser = argparse.ArgumentParser(description='Console programm for test with loss function.')

    parser.add_argument('--original_image', type=str, required=True, 
                        help='Original image.')
    parser.add_argument('--distorted_image', type=str, required=True, 
                        help='Distorted image.')

    parser.add_argument('--gen_distortion', action='store_true', default=False,
                        help='Generate distorted image with given kernel.')

    parser.add_argument('--xp', type=int, nargs='+', default=[1, 5, 11], 
                        help='x coordinates of points of kernel')
    parser.add_argument('--yp', type=int, nargs='+', default=[8, 4, 9], 
                        help='y coordinates of points of kernel')
    parser.add_argument('--kernel_size', type=int, default=11, help='Size of generated kernel.')

    parser.add_argument('--loss', type=str, nargs='+', default=['L2'], 
                        choices=AVAILABLE_LOSSES, help='Used loss function.')

    parser.add_argument('--gray', action='store_true', default=False, 
                        help='If set, loss will be optimized with grayscale images.')

    return parser.parse_args()



def show_dump(dump):
    for key, val in dump.items():
        plt.figure()
        plt.imshow(val, 'gray')
        plt.title(key)
    plt.show()


def main():
    args = get_args()

    if not args.gray:
        raise NotImplementedError('Work with color images is not implemented yet.')

    dump = {}

    original_image = i2t(read_image(args.original_image, args.gray))

    if args.gen_distortion:
        assert len(args.xp) == len(args.yp) == 3
        generated_kernel = generate_kernel(np.array(args.xp), np.array(args.yp), args.kernel_size)
        if generated_kernel is None:
            print('Poor conditioned kernel.')
            exit(-1)

        dump['generated_kernel'] = generated_kernel

        distorted_image = conv(original_image, generated_kernel)
    else:
        distorted_image = i2t(read_image(args.distorted_image, args.gray))

    dump['original_image'] = t2i(original_image)
    dump['distorted_image'] = t2i(distorted_image)

    criteria = Loss(args.loss)

    kernel, kernel_size, kernel_angle = initialize_kernel(t2i(distorted_image))
    kernel = kernel_threshold(kernel)
    print('ESTIMATED KERNEL PARAMETERS: size - {} | angle - {}'.format(kernel_size, kernel_angle))

    dump['estimated_kernel'] = kernel

    m_kernel = MutableKernel(kernel)

    losses = np.zeros((m_kernel._kernel_size, m_kernel._kernel_size))
    kernels = np.zeros((m_kernel._kernel_size ** 2, m_kernel._kernel_size ** 2))

    for x, y, cur_kernel in m_kernel:
        if cur_kernel is not None: 
            loss_v = criteria(distorted_image, conv(original_image, cur_kernel))
            print('LOSS {}: {}'.format(criteria, loss_v))
            kernels[y * m_kernel._kernel_size:(y+1) * m_kernel._kernel_size, \
                    x * m_kernel._kernel_size:(x+1) * m_kernel._kernel_size] = cur_kernel
            losses[y, x] = loss_v

    dump['losses'] = losses
    dump['kernels'] = kernels

    show_dump(dump)


if __name__ == '__main__':
    main()
