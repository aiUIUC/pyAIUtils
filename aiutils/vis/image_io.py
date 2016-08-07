from PIL import Image
import numpy as np


def imread(filename):
    """
    Matlab like function for reading an image file.

    Returns:
    im_array (numpy.ndarray): h x w x 3 ndarray for color images and 
        h x w for grayscale images
    """
    im = Image.open(filename)

    err_str = \
        "imread only supports 'RGB' and 'L' modes, found '{}'".format(im.mode)
    assert (im.mode == 'RGB' or im.mode == 'L'), err_str

    im_array = np.array(im)

    return im_array


def imshow(np_im):
    """
    Matlab like function for displaying a numpy ndarray as an image

    Args:
    np_im (numpy.ndarray): h x w x 3 ndarray for color images and 
        h x w for grayscale images with pixels stored in uint8 format
    """
    err_str = 'imshow expects ndarray of dimension h x w x c (RGB) or h x w (L)'
    assert (len(np_im.shape) == 3 or len(np_im.shape) == 2), err_str

    if len(np_im.shape) == 3:
        assert (np_im.shape[2] == 3), 'imshow expected 3 channels'
        im = Image.fromarray(np_im, 'RGB')
    else:
        im = Image.fromarray(np_im, 'L')

    im.show()


def imwrite(np_im, filename):
    """
    Matlab like function for displaying a numpy ndarray as an image

    Args:
    np_im (numpy.ndarray): h x w x 3 ndarray for color images and 
        h x w for grayscale images with pixels stored in uint8 format
    """
    err_str = 'imwrite expects ndarray of dimension h x w x c (RGB) or h x w (L)'
    assert (len(np_im.shape) == 3 or len(np_im.shape) == 2), err_str

    assert(np_im.dtype == np.dtype('uint8')), 'expects np_im to be a uint8, 0-255 valued'

    if len(np_im.shape) == 3:
        assert (np_im.shape[2] == 3), 'imwrite expected 3 channels'
        im = Image.fromarray(np_im, 'RGB')
    else:
        im = Image.fromarray(np_im, 'L')

    im.save(filename)
