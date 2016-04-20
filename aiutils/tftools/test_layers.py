import numpy as np
import tensorflow as tf

import layers


def test_full():
    batch = 32
    in_dim = 3
    out_dim = 10

    input_shape = [batch, in_dim]
    output_shape = [batch, out_dim]

    x = tf.placeholder(tf.float32, input_shape)
    y = layers.full(x, out_dim, 'full')
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    x_ = np.float32(np.zeros(input_shape))
    y_ = np.float32(np.zeros(output_shape))
    y_hat = sess.run(y, feed_dict={x: x_})
    assert (np.all(y_hat == y_))


def test_conv2d():
    batch = 32
    height = 64
    width = 64
    in_dim = 3
    out_dim = 10

    input_shape = [batch, height, width, in_dim]
    output_shape = [batch, height, width, out_dim]

    x = tf.placeholder(tf.float32, input_shape)
    y = layers.conv2d(x, in_dim, out_dim, 'conv2d')
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    x_ = np.float32(np.zeros(input_shape))
    y_ = np.float32(np.zeros(output_shape))
    y_hat = sess.run(y, feed_dict={x: x_})
    assert (np.all(y_hat == y_))
