import argparse

import numpy as np
import torch

from config import *
from convolve import conv
from kerneltools.init_kernel import initialize_kernel, kernel_threshold
from iom import read_image, write_image
from loss import Loss
from mutable_kernel import MutableKernel
from utils import i2t, t2i


def get_args():
    parser = argparse.ArgumentParser(description='Console programm for test with loss function.')

    parser.add_argument('--original_image', type=str, required=True, 
                        help='Original image.')
    parser.add_argument('--distorted_image', type=str, required=True, 
                        help='Distorted image.')

    parser.add_argument('--loss', type=str, nargs='+', default=['L2'], 
                        choices=AVAILABLE_LOSSES, help='Used loss function.')

    parser.add_argument('--gray', action='store_true', default=False, 
                        help='If set, loss will be optimized with grayscale images.')

    return parser.parse_args()


def main():
    args = get_args()

    if not args.gray:
        raise NotImplementedError('Work with color images is not implemented yet.')

    original_image = i2t(read_image(args.original_image, args.gray))
    distorted_image = i2t(read_image(args.distorted_image, args.gray))

    criteria = Loss(args.loss)

    kernel, kernel_size, kernel_angle = initialize_kernel(t2i(distorted_image))
    kernel = kernel_threshold(kernel)
    print('ESTIMATED KERNEL PARAMETERS: size - {} | angle - {}'.format(kernel_size, kernel_angle))

    for cur_kernel in MutableKernel(kernel):
        if cur_kernel is not None: 
            print('LOSS {}: {}'.format(criteria, criteria(distorted_image, conv(original_image, cur_kernel))))


if __name__ == '__main__':
    main()
