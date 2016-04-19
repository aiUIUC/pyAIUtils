from termcolor import colored
from PIL import Image
import numpy as np
import pdb


def raise_exception(e, err_str):
    pkg_err_str = 'Exception raised in package pyAIUtil: \n{}'.format(err_str)
    print colored(pkg_err_str, 'red')
    raise


def check_assertion(cond, err_str):
    pkg_err_str = '\nAssertion failed in package pyAIUtil: \n{}'.format(
        err_str)
    assert (cond), colored(pkg_err_str, 'blue')


def imread(filename):
    try:
        im = Image.open(filename)

    except IOError as e:
        err_str = 'imread could not open file {}'.format(filename)
        raise_exception(e, err_str)

    assert_cond = im.mode == 'RGB' or im.mode == 'L'
    err_str = \
        "imread only supports 'RGB' and 'L' modes, found '{}'".format(im.mode)
    check_assertion(assert_cond, err_str)

    im_seq = np.array(im.getdata(), np.uint8)

    if im.mode == 'RGB':
        np_im = im_seq.reshape(im.size[1], im.size[0], 3)

    elif im.mode == 'L':
        np_im = im_seq.reshape(im.size[1], im.size[0])

    return np_im


def imshow(np_im):
    assert_cond = len(np_im.shape)==3 or len(np_im.shape)==2
    err_str = 'imshow expects ndarray of dimension h x w x c (RGB) or h x w (L)'
    check_assertion(assert_cond, err_str)
    
    if len(np_im.shape) == 3:
        check_assertion(np_im.shape[2]==3, 'imshow expected 3 channels')
        im = Image.fromarray(np_im, 'RGB')

    else:
        im = Image.fromarray(np_im, 'L')

    im.show()


