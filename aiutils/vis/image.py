from PIL import Image
import numpy as np
import math


def rgb2gray(np_im):
    """
    Matlab like function for converting numpy ndarray from RGB to grayscale

    Args:
    np_im (numpy.ndarray): h x w x 3 ndarray storing a color image
    """
    err_str = 'rgb2gray expects ndarray of dimension h x w x 3 (RGB)'
    assert (len(np_im.shape) == 3), err_str

    im = Image.fromarray(np_im, 'RGB')
    im_array = np.array(im.convert('L'))

    return im_array


def imresize(np_im, method='bilinear', **kwargs):
    """
    Matlab like function for resizing image stored as numpy ndarray.

    Args:
    np_im (numpy.ndarray): h x w x 3 ndarray for color images and 
        h x w for grayscale images with pixels stored in uint8 format
    method: Algorithm to use for interpolation. Must be one of
        {'bilinear', 'nearest', 'lanczos'}

    output_size (int list): output size stored in a list as [h, w]
    scale (float): scaling factor by which to resize the image. Use 
        either output_size or scale. Exception is thrown otherwise
    """
    assert_str = "Only 1 keyword argument expected with key either" + \
                 "'output_size' or 'scale'"
    assert (len(kwargs) == 1), assert_str

    if method == 'bilinear':
        method_ = Image.BILINEAR
    elif method == 'nearest':
        method_ = Image.NEAREST
    elif method == 'lanczos':
        method_ = Image.LANCZOS
    else:
        assert_str = "Interpolation method must be one of " + \
                     "{'bilinear', 'nearest', 'lanczos'}"
        assert (False), assert_str

    im_h = np_im.shape[0]
    im_w = np_im.shape[1]
    if 'output_size' in kwargs:
        h, w = kwargs['output_size']
    elif 'scale' in kwargs:
        h = scale * im_h
        w = scale * im_w
    else:
        assert_str = "Variable argument must be one of {'output_size','scale'}"
        assert (False), assert_str
    h = int(math.ceil(h))
    w = int(math.ceil(w))
    im = Image.fromarray(np_im)
    return np.array(im.resize((w, h), resample=method_))
