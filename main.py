import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import *
from convolve import conv
from iom import *
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

    parser.add_argument('--lr_itr', type=int, default=30, help='Number of iterations for RL algorithm.')

    parser.add_argument('--init_with_grid', action='store_true', default=False, 
                        help='Find estimation of kernel using grid search.')

    parser.add_argument('--dump_path', type=str, default=None, 
                        help='Save dump to this file.')

    return parser.parse_args()



def process_dump(dump_path, dump):
    for key, val in dump.items():
        if isinstance(val, np.ndarray):
            plt.figure()
            plt.imshow(val, 'gray')
            plt.title(key)

    dump_mat(dump_path, dump)

    plt.show()


def grid_search(original_image, distorted_image, m_kernel, criteria, save=False):
    min_loss_v = np.inf
    kernel, points = None, None

    all_data = [] if save else None

    for cur_points, cur_kernel in m_kernel:
        if cur_kernel is not None: 
            loss_v = criteria(conv(original_image, cur_kernel), distorted_image)
            print('Points: {} | Loss {}: {}'.format(cur_points, criteria, loss_v))

            if loss_v < min_loss_v:
                min_loss_v, kernel, points = loss_v, cur_kernel, cur_points

            if save:
                all_data.append((cur_kernel, cur_points, loss_v))

    return kernel, points, all_data


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

    if args.init_with_grid:
        m_kernel = MutableKernel(kernel_size, order=1)
        save = True if args.dump_path is not None else False
        kernel, _ , _ = grid_search(original_image, distorted_image, m_kernel, criteria)

    dump['estimated_kernel'] = kernel
    dump['original_image'] = t2i(original_image)
    dump['distorted_image'] = t2i(distorted_image)

    if args.grid_search:
        m_kernel = MutableKernel(kernel_size, kernel)
        save = True if args.dump_path is not None else False
        found_kernel, _, all_data = grid_search(original_image, distorted_image, m_kernel, criteria, 
                                         save=save)
        dump['found_kernel'] = found_kernel
        if save:
            dump['all_data'] = all_data

    else:
        grad_desc(original_image, distorted_image, kernel, criteria, dump)

    dump['restored_image'] = restoration.richardson_lucy(t2i(distorted_image), 
                                                         dump['found_kernel'], args.lr_itr)

    process_dump(args.dump_path, dump)


if __name__ == '__main__':
    main()
