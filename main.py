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

from skimage import restoration


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

    parser.add_argument('--grid_search', action='store_true', default=False, 
                        help='Find distortion kernel using grid search instead gradient descent.')

    parser.add_argument('--lr_itr', type=int, default=10, help='Number of iterations for RL algorithm.')

    return parser.parse_args()



def show_dump(dump):
    for key, val in dump.items():
        plt.figure()
        plt.imshow(val, 'gray')
        plt.title(key)
    plt.show()


def grid_search(original_image, distorted_image, init_kernel, criteria, dump):
    m_kernel = MutableKernel(init_kernel)

    losses = np.inf * np.ones((m_kernel._kernel_size, m_kernel._kernel_size))
    kernels = np.zeros((m_kernel._kernel_size ** 2, m_kernel._kernel_size ** 2))

    for x, y, cur_kernel in m_kernel:
        if cur_kernel is not None: 
            loss_v = criteria(conv(original_image, cur_kernel), distorted_image)
            print('LOSS {}: {}'.format(criteria, loss_v))
            kernels[y * m_kernel._kernel_size:(y+1) * m_kernel._kernel_size, \
                    x * m_kernel._kernel_size:(x+1) * m_kernel._kernel_size] = cur_kernel
            losses[y, x] = loss_v

    yi, xi = np.where(losses == np.min(losses))
    yi, xi = yi[0], xi[0]
    print('Minimum value: {} | xi: {} | yi: {}'.format(losses[yi, xi], xi, yi))
    dump['found_kernel'] = kernels[yi * m_kernel._kernel_size:(yi+1) * m_kernel._kernel_size, \
                                    xi * m_kernel._kernel_size:(xi+1) * m_kernel._kernel_size]

    dump['losses'] = losses
    dump['kernels'] = kernels


def grad_desc(original_image, distorted_image, init_kernel, criteria, dump):
    kernel = np.expand_dims(init_kernel, axis=0)
    kernel = np.expand_dims(kernel, axis=0)
    kernel = torch.tensor(kernel.astype(np.float32), requires_grad=True)

    optimizer = torch.optim.Adam([kernel], 0.5)

    for i in range(1000):
        optimizer.zero_grad()
        
        loss_v = criteria(conv(original_image, kernel, True), distorted_image, True) + 0.01 * torch.mean(kernel ** 2)
        loss_v.backward()
        optimizer.step()

        print('I: {} | LOSS: {:.10f}'.format(i, loss_v))

        # kernel.data /= torch.sum(kernel.data)

    dump['found_kernel'] = t2i(kernel)


def main():
    args = get_args()

    if not args.gray:
        raise NotImplementedError('Work with color images is not implemented yet.')

    dump = {}

    original_image = i2t(read_image(args.original_image, args.gray))

    if args.gen_distortion:
        assert len(args.xp) == len(args.yp)
        generated_kernel = generate_kernel(np.array(args.xp), np.array(args.yp), args.kernel_size)
        if generated_kernel is None:
            print('Poor conditioned kernel.')
            exit(-1)

        dump['generated_kernel'] = generated_kernel

        distorted_image = conv(original_image, generated_kernel)
    else:
        distorted_image = i2t(read_image(args.distorted_image, args.gray))

    criteria = Loss(args.loss)

    kernel, kernel_size, kernel_angle = initialize_kernel(t2i(distorted_image))
    kernel = kernel_threshold(kernel)
    print('ESTIMATED KERNEL PARAMETERS: size - {} | angle - {}'.format(kernel_size, kernel_angle))

    dump['estimated_kernel'] = kernel
    dump['original_image'] = t2i(original_image)
    dump['distorted_image'] = t2i(distorted_image)

    if args.grid_search:
        grid_search(original_image, distorted_image, kernel, criteria, dump)

    else:
        grad_desc(original_image, distorted_image, kernel, criteria, dump)

    dump['restored_image'] = restoration.richardson_lucy(t2i(distorted_image), 
                                                         dump['found_kernel'], args.lr_itr)

    show_dump(dump)


if __name__ == '__main__':
    main()
