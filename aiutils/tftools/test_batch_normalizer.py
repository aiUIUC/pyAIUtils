import numpy as np
import pytest
import tensorflow as tf

import batch_normalizer
from batch_normalizer import BatchNormalizer
from batch_normalizer import batch_normalize


@pytest.fixture(scope='function')
def graph(request):
    g = tf.Graph()
    sess = tf.Session(graph=g)

    def teardown():
        sess.close()

    request.addfinalizer(teardown)

    return (g, sess)


def test_batch_normalize_construct_2(graph):
    g = graph[0]
    with g.as_default():
        input_shape = [1, 10]
        x = tf.placeholder(tf.float32, input_shape)
        ema = tf.train.ExponentialMovingAverage(decay=.9)
        bn = BatchNormalizer(x, ema, "batch_norm1")
        assert (bn)


def test_batch_normalize_construct_4(graph):
    g = graph[0]
    with g.as_default():
        input_shape = [1, 10, 5, 8]
        x = tf.placeholder(tf.float32, input_shape)
        ema = tf.train.ExponentialMovingAverage(decay=.9)
        bn = BatchNormalizer(x, ema, "batch_norm2")
        assert (bn)


def test_batch_normalize_construct_3(graph):
    g = graph[0]
    with g.as_default():
        with pytest.raises(ValueError):
            input_shape = [1, 10, 8]
            x = tf.placeholder(tf.float32, input_shape)
            ema = tf.train.ExponentialMovingAverage(decay=.9)
            bn = BatchNormalizer(x, ema, "batch_norm3")


def test_batch_normalize(graph):
    g = graph[0]
    with g.as_default():
        input_shape = [5, 10]
        x = tf.placeholder(tf.float32, input_shape)
        ema = tf.train.ExponentialMovingAverage(decay=.9)
        bn = BatchNormalizer(x, ema, "batch_norm")

        x_val = np.random.rand(*input_shape)
        normalize = bn.normalize(x)

        sess = graph[1]
        sess.run(tf.initialize_all_variables())
        y = sess.run(normalize, {x: x_val})
        assert (y.shape == x_val.shape)


def test_batch_normalize_fcn(graph):
    g = graph[0]
    with g.as_default():
        input_shape = [5, 10]
        x = tf.placeholder(tf.float32, input_shape)
        phase_train = tf.placeholder(tf.bool)

        normalize = batch_normalize(x, phase_train, 'batch_norm4')

        assert (len(tf.get_collection('batch_norm_average_update')) == 1)

        x_val = np.random.rand(*input_shape)

        sess = graph[1]
        sess.run(tf.initialize_all_variables())
        # Just verify that this doesn't error, not really looking for any specific results
        y1 = sess.run(normalize, {x: x_val, phase_train: True})
        y2 = sess.run(normalize, {x: x_val, phase_train: False})
