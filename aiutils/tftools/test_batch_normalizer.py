import numpy as np
import pytest
import tensorflow as tf

import batch_normalizer
from batch_normalizer import BatchNormalizer
from batch_normalizer import batch_normalize, BatchNorm


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

def test_batchnorm_train_mode():
    batch = 5
    width = 2
    height = 3
    channels = 4

    input_shape = [batch, width, height, channels]

    training = True
    x = tf.placeholder(tf.float32, input_shape)

    bn = BatchNorm(x, training, name='bn')
    y = bn.output
    ema_mean, ema_var = bn.get_ema_moments()

    sess = tf.Session()

    x_val1 = np.ones(input_shape, dtype=np.float32)
    x_val2 = 2.0 * x_val1

    with sess.as_default():
        sess.run(tf.initialize_all_variables())

        y_eval1 = y.eval(feed_dict={x: x_val1})
        ema_mean_eval1 = ema_mean.eval()

        y_eval2 = y.eval(feed_dict={x: x_val2})
        ema_mean_eval2 = ema_mean.eval()

    sess.close()
    tf.reset_default_graph()

    assert_str = 'batch mean and var are not used correctly' + \
                 'during training with batch norm'
    assert (np.all(y_eval1 == np.zeros(input_shape))), assert_str
    assert_str = 'batch mean and var are not used correctly' + \
                 'during training with batch norm'
    assert (np.all(y_eval2 == np.zeros(input_shape))), assert_str
    assert_str = 'ema mean is not updated during training with batch norm'
    assert (not np.all(ema_mean_eval1 == ema_mean_eval2)), assert_str


def test_batchnorm_test_mode():
    batch = 5
    width = 2
    height = 3
    channels = 4

    input_shape = [batch, width, height, channels]

    training = False
    x = tf.placeholder(tf.float32, input_shape)

    bn = BatchNorm(x, training, name='bn')
    y = bn.output
    ema_mean, ema_var = bn.get_ema_moments()

    sess = tf.Session()

    x_val1 = np.ones(input_shape, dtype=np.float32)
    x_val2 = 2.0 * x_val1

    with sess.as_default():
        sess.run(tf.initialize_all_variables())

        y_eval1 = y.eval(feed_dict={x: x_val1})
        ema_mean_eval1 = ema_mean.eval()

        y_eval2 = y.eval(feed_dict={x: x_val2})
        ema_mean_eval2 = ema_mean.eval()

    sess.close()
    tf.reset_default_graph()

    assert_str = 'ema mean and var are not used correctly' + \
                 'during testing with batch norm'
    assert (not np.all(y_eval1 == np.zeros(input_shape))), assert_str
    assert_str = 'ema mean and var are not used correctly' + \
                 'during testing with batch norm'
    assert (not np.all(y_eval2 == np.zeros(input_shape))), assert_str
    assert_str = 'ema mean is updated during testing with batch norm'
    assert (np.all(ema_mean_eval1 == ema_mean_eval2)), assert_str
