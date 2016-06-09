import numpy as np
import pytest
import tensorflow as tf

from batch_normalizer import BatchNorm
from layers import batch_norm

def test_batchnorm_train_mode():
    batch = 5
    width = 2
    height = 3
    channels = 4

    input_shape = [batch, width, height, channels]

    g = tf.Graph()
    with g.as_default():
        training = tf.placeholder(tf.bool, [])
        x = tf.placeholder(tf.float32, input_shape)
        bn = BatchNorm(x, training, name='bn')
        y = bn.output
        ema_mean, ema_var = bn.get_ema_moments()
        initializer = tf.initialize_all_variables()

    x_val1 = np.ones(input_shape, dtype=np.float32)
    x_val2 = 2.0 * x_val1

    sess = tf.Session(graph=g)
    with sess.as_default():
        sess.run(initializer)
        y_eval1 = y.eval(feed_dict={x: x_val1, training: True})
        ema_mean_eval1 = ema_mean.eval()

        y_eval2 = y.eval(feed_dict={x: x_val2, training: True})
        ema_mean_eval2 = ema_mean.eval()

    sess.close()

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

    g = tf.Graph()
    with g.as_default():
        training = tf.placeholder(tf.bool, [])
        x = tf.placeholder(tf.float32, input_shape)
        bn = BatchNorm(x, training, name='bn')
        y = bn.output
        ema_mean, ema_var = bn.get_ema_moments()
        initializer = tf.initialize_all_variables()

    x_val1 = np.ones(input_shape, dtype=np.float32)
    x_val2 = 2.0 * x_val1

    sess = tf.Session(graph=g)
    with sess.as_default():
        sess.run(initializer)

        y_eval1 = y.eval(feed_dict={x: x_val1, training: False})
        ema_mean_eval1 = ema_mean.eval()

        y_eval2 = y.eval(feed_dict={x: x_val2, training: False})
        ema_mean_eval2 = ema_mean.eval()

    sess.close()

    assert_str = 'ema mean and var are not used correctly' + \
                 'during testing with batch norm'
    assert (not np.all(y_eval1 == np.zeros(input_shape))), assert_str
    assert_str = 'ema mean and var are not used correctly' + \
                 'during testing with batch norm'
    assert (not np.all(y_eval2 == np.zeros(input_shape))), assert_str
    assert_str = 'ema mean is updated during testing with batch norm'
    assert (np.all(ema_mean_eval1 == ema_mean_eval2)), assert_str

if __name__=='__main__':
    test_batchnorm_train_mode()
    test_batchnorm_test_mode()
