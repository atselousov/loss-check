from skimage import io
import scipy.io
import numpy as np


def read_image(path, gray=False):
    image = io.imread(path, as_grey=gray).astype(np.float32)

    image = image / 255.
    
    return image


def write_image(path, image):
    # if len(image.shape) != 2:
    image = image * 255.

    image = image.astype(np.uint8)

    io.imsave(path, image)


def dump_mat(path, dict):
    if path is not None:
        scipy.io.savemat(path, dict)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    image = read_image('https://img-9gag-fun.9cache.com/photo/am7ppD2_700b.jpg')
    print(image.shape)
    print('MAX: {} | MIN: {}'.format(np.max(image), np.min(image)))

    write_image('test.png', image)

    plt.imshow(np.squeeze(image))
    plt.show()
