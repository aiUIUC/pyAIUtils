import numpy
import tensorflow as tf


def identity(x):
    return x


def full(x, out_dim, name, gain=numpy.sqrt(2), func=tf.nn.relu):
    in_dim = x.get_shape().as_list()[-1]
    stddev = 1.0 * gain / numpy.sqrt(in_dim)
    with tf.variable_scope(name):
        w_init = tf.random_normal_initializer(stddev)
        b_init = tf.constant_initializer()
        w = tf.get_variable('w',
                            shape=[in_dim, out_dim],
                            initializer=w_init)
        b = tf.get_variable('b',
                            shape=[out_dim],
                            initializer=b_init)
        out = func(tf.matmul(x, w) + b)
    return out