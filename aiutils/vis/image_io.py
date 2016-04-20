from PIL import Image
import numpy as np
import pdb

import aiutils.devtools.err_msg as err_msg


def imread(filename):
    """
    Matlab like function for reading an image file.

    Returns:
    np_im (numpy.ndarray): h x w x 3 ndarray for color images and 
        h x w for grayscale images
    """
    try:
        im = Image.open(filename)
    except IOError as e:
        err_str = 'imread could not open file {}'.format(filename)
        err_msg.raise_exception(e, err_str)

    assert_cond = im.mode == 'RGB' or im.mode == 'L'
    err_str = \
        "imread only supports 'RGB' and 'L' modes, found '{}'".format(im.mode)
    err_msg.check_assertion(assert_cond, err_str)

    im_seq = np.array(im.getdata(), np.uint8)

    if im.mode == 'RGB':
        np_im = im_seq.reshape(im.size[1], im.size[0], 3)
    elif im.mode == 'L':
        np_im = im_seq.reshape(im.size[1], im.size[0])

    return np_im


def imshow(np_im):
    """
    Matlab like function for displaying a numpy ndarray as an image

    Args:
    np_im (numpy.ndarray): h x w x 3 ndarray for color images and 
        h x w for grayscale images with pixels stored in uint8 format
    """
    assert_cond = len(np_im.shape) == 3 or len(np_im.shape) == 2
    err_str = 'imshow expects ndarray of dimension h x w x c (RGB) or h x w (L)'
    err_msg.check_assertion(assert_cond, err_str)

    if len(np_im.shape) == 3:
        err_msg.check_assertion(np_im.shape[2] == 3,
                                'imshow expected 3 channels')
        im = Image.fromarray(np_im, 'RGB')
    else:
        im = Image.fromarray(np_im, 'L')

    im.show()


if __name__=='__main__':
    filename = '/home/tanmay/Desktop/peppers.png'
    im = imread(filename)
    imshow(im)
