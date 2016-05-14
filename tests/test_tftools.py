import numpy as np
import scipy.sparse as sps
import pytest
import tensorflow as tf

from context import layers
from context import images
from context import plholder_management


def test_full():
    batch = 1
    in_dim = 2
    out_dim = 3

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
    batch = 1
    height = 3
    width = 3
    filter_size = 3
    in_dim = 4
    out_dim = 5

    input_shape = [batch, height, width, in_dim]
    output_shape = [batch, height, width, out_dim]

    x = tf.placeholder(tf.float32, input_shape)
    y = layers.conv2d(x, filter_size, out_dim, 'conv2d')
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    x_ = np.float32(np.zeros(input_shape))
    y_ = np.float32(np.zeros(output_shape))
    y_hat = sess.run(y, feed_dict={x: x_})
    assert (np.all(y_hat == y_))


def test_batch_norm_2d():
    batch = 1
    in_dim = 2
    out_dim = 3

    input_shape = [batch, in_dim]

    x = tf.placeholder(tf.float32, input_shape)
    y = layers.batch_norm(x, '2d')
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    x_ = np.float32(np.random.randn(*input_shape))
    y_hat = sess.run(y, feed_dict={x: x_})

    assert y_hat.shape == x_.shape


def test_batch_norm_4d():
    batch = 1
    width = 2
    height = 3
    in_dim = 4
    out_dim = 5

    input_shape = [batch, width, height, in_dim]

    x = tf.placeholder(tf.float32, input_shape)
    y = layers.batch_norm(x, '4d')
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    x_ = np.float32(np.random.randn(*input_shape))
    y_hat = sess.run(y, feed_dict={x: x_})

    assert y_hat.shape == x_.shape


def test_batch_norm_3d():
    batch = 1
    width = 2
    in_dim = 3
    out_dim = 4

    input_shape = [batch, width, in_dim]

    x = tf.placeholder(tf.float32, input_shape)
    with pytest.raises(ValueError):
        y = layers.batch_norm(x, '4d')


def test_resize_image_like():
    batch = 1
    width_height = 5
    width_height2 = 15
    channel = 3

    input_shape = [batch, width_height, width_height, channel]
    target_shape = [batch, width_height2, width_height2, channel]

    x = tf.placeholder(tf.float32, input_shape)
    s = tf.placeholder(tf.float32, target_shape)
    x_resize = images.resize_images_like(x, s)
    assert x_resize.get_shape() == s.get_shape()

    x_resize2 = images.resize_images_like(x, s, method=1)
    assert x_resize2.get_shape() == s.get_shape()

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    x_ = np.float32(np.random.rand(*input_shape))
    #batch, width_height, width_height, channel))
    x_out = sess.run(x_resize, {x: x_})
    assert x_out.shape == s.get_shape()


def test_plholder_management():
    sess = tf.InteractiveSession()

    plh_mgr = plholder_management.PlholderManager()

    # Add placeholders
    plh_mgr.add_plholder('word_embed', tf.float64, [10, 10])
    plh_mgr.add_plholder('sp_ids', tf.int64, sparse=True)
    plh_mgr.add_plholder('weights', tf.float64, sparse=True)

    # Get a dictionary of placeholders
    plhs = plh_mgr.get_plholders()

    # Define computation graph
    y = tf.nn.embedding_lookup_sparse(plhs['word_embed'], plhs['sp_ids'],
                                      plhs['weights'])

    # Create data to be fed into the graph
    I = np.array([0, 0, 1, 1])
    J = np.array([3, 8, 2, 5])
    V = np.array([3, 8, 2, 5])
    W = np.array([0.9, 0.1, 0.4, 0.6])
    sp_ids = sps.coo_matrix((V, (I, J)), shape=(2, 10), dtype=np.int64)
    weights = sps.coo_matrix((W, (I, J)), shape=(2, 10), dtype=np.float64)
    word_embed = np.eye(10, 10, dtype=np.float64)

    # Create input dict
    inputs = {'word_embed': word_embed, 'sp_ids': sp_ids, 'weights': weights, }

    # Create feed dictionary from inputs
    feed_dict = plh_mgr.get_feed_dict(inputs)

    y_gt = np.array([[0, 0, 0, 0.9, 0, 0, 0, 0, 0.1, 0], \
                     [0, 0, 0.4, 0, 0, 0.6, 0, 0, 0, 0]])

    assert_str = 'test_plholder_management failed'
    assert (np.array_equal(y.eval(feed_dict), y_gt)), assert_str

    assert_str = '__getitem__ method of PlholderManager class failed'
    assert (plh_mgr['word_embed'].name=='word_embed'), assert_str
