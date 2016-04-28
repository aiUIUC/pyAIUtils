import numpy as np
import pytest
import tensorflow as tf

import batch_normalizer
from batch_normalizer import BatchNormalizer
from batch_normalizer import batch_normalize



def test_batch_normalize_construct_2():
    input_shape = [1, 10]
    x = tf.placeholder(tf.float32, input_shape)
    ema = tf.train.ExponentialMovingAverage(decay=.9)
    bn = BatchNormalizer(x, ema, "batch_norm1")
    assert(bn)

def test_batch_normalize_construct_4():
    input_shape = [1, 10, 5, 8]
    x = tf.placeholder(tf.float32, input_shape)
    ema = tf.train.ExponentialMovingAverage(decay=.9)
    bn = BatchNormalizer(x, ema, "batch_norm2")
    assert(bn)

def test_batch_normalize_construct_3():
    with pytest.raises(ValueError):
        input_shape = [1, 10, 8]
        x = tf.placeholder(tf.float32, input_shape)
        ema = tf.train.ExponentialMovingAverage(decay=.9)
        bn = BatchNormalizer(x, ema, "batch_norm3")


def test_batch_normalize():
    input_shape = [5, 10]
    x = tf.placeholder(tf.float32, input_shape)
    ema = tf.train.ExponentialMovingAverage(decay=.9)
    bn = BatchNormalizer(x, ema, "batch_norm")

    x_val = np.random.rand(*input_shape)
    normalize = bn.normalize(x)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    y = sess.run(normalize, {x:x_val})
    assert(y.shape == x_val.shape)

def test_batch_normalize_fcn():
    input_shape = [5, 10]
    x = tf.placeholder(tf.float32, input_shape)
    phase_train = tf.placeholder(tf.bool)

    batch_normalize(x, phase_train, 'batch_norm4')

    assert(len(tf.get_collection('batch_norm_average_update'))==1)
