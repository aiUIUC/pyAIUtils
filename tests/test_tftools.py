import numpy as np
import scipy.sparse as sps
import pytest
import tensorflow as tf

from context import layers
from context import images
from context import placeholder_management
from context import batch_normalizer
from context import var_collect


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

    sess.close()
    tf.reset_default_graph()


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

    sess.close()
    tf.reset_default_graph()


def test_atrous_conv2d():

    input_shape = [10, 100, 100, 3]
    filter_shape = 5

    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(tf.float32, input_shape)
        a1 = layers.atrous_conv2d(x, 5, 8, 'atrous1')
        a2 = layers.atrous_conv2d(x, 5, 8, 'atrous2', rate=5)

        assert (a1.get_shape() == a2.get_shape())
        assert (int(a1.get_shape()[0]) == 10)
        assert (int(a1.get_shape()[1]) == 100)
        assert (int(a1.get_shape()[2]) == 100)
        assert (int(a1.get_shape()[3]) == 8)


def test_batch_norm_2d():
    batch = 1
    in_dim = 2
    out_dim = 3

    input_shape = [batch, in_dim]

    x = tf.placeholder(tf.float32, input_shape)
    training = tf.placeholder(tf.bool, [])
    y = layers.batch_norm(x, training)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    x_ = np.float32(np.random.randn(*input_shape))
    y_hat = sess.run(y, feed_dict={x: x_, training: True})

    assert y_hat.shape == x_.shape

    sess.close()
    tf.reset_default_graph()


def test_batch_norm_4d():
    batch = 1
    width = 2
    height = 3
    in_dim = 4
    out_dim = 5

    input_shape = [batch, width, height, in_dim]

    x = tf.placeholder(tf.float32, input_shape)
    training = tf.placeholder(tf.bool, [])
    y = layers.batch_norm(x, training)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    x_ = np.float32(np.random.randn(*input_shape))
    y_hat = sess.run(y, feed_dict={x: x_, training: True})

    assert y_hat.shape == x_.shape
    sess.close()
    tf.reset_default_graph()


def test_batch_norm_3d():
    batch = 1
    width = 2
    in_dim = 3
    out_dim = 4

    input_shape = [batch, width, in_dim]

    x = tf.placeholder(tf.float32, input_shape)
    training = tf.placeholder(tf.bool, [])
    with pytest.raises(ValueError):
        y = layers.batch_norm(x, training)

    tf.reset_default_graph()


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
        bn = batch_normalizer.BatchNorm(x, training, name='bn')
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
        bn = batch_normalizer.BatchNorm(x, training, name='bn')
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


def test_dropout():
    input_shape = [10, 6]

    g = tf.Graph()
    with g.as_default():
        training = tf.placeholder(tf.bool, [])
        inp = tf.placeholder(tf.float32, input_shape)

        x = layers.dropout(inp, True)
        assert (x.get_shape() == inp.get_shape())
        x = layers.dropout(inp, False)
        assert (x.get_shape() == inp.get_shape())
        x = layers.dropout(inp, training)
        assert (x.get_shape() == inp.get_shape())
        with pytest.raises(TypeError):
            x = layers.dropout(inp, 10)
        with pytest.raises(TypeError):
            x = layers.dropout(inp, inp)


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
    x_out = sess.run(x_resize, {x: x_})
    assert x_out.shape == s.get_shape()

    sess.close()
    tf.reset_default_graph()


def test_placeholder_management():
    sess = tf.InteractiveSession()

    plh_mgr = placeholder_management.PlaceholderManager()

    # Add placeholders
    plh_mgr.add_placeholder('word_embed', tf.float64, [10, 10])
    plh_mgr.add_placeholder('sp_ids', tf.int64, sparse=True)
    plh_mgr.add_placeholder('weights', tf.float64, sparse=True)

    # Get a dictionary of placeholders
    plhs = plh_mgr.get_placeholders()

    # Define computation graph
    y = tf.nn.embedding_lookup_sparse(plhs['word_embed'], plhs['sp_ids'],
                                      plhs['weights'])

    # Create data to be fed into the graph
    I = np.array([0, 0, 1, 1])
    J = np.array([0, 1, 2, 3])
    V = np.array([3, 8, 2, 5])
    W = np.array([0.9, 0.1, 0.4, 0.6])
    sp_ids = sps.coo_matrix((V, (I, J)), shape=(2, 4), dtype=np.int64)
    weights = sps.coo_matrix((W, (I, J)), shape=(2, 4), dtype=np.float64)
    word_embed = np.eye(10, 10, dtype=np.float64)

    # Create input dict
    inputs = {'word_embed': word_embed, 'sp_ids': sp_ids, 'weights': weights, }

    # Create feed dictionary from inputs
    feed_dict = plh_mgr.get_feed_dict(inputs)

    y_gt = np.array([[0, 0, 0, 0.9, 0, 0, 0, 0, 0.1, 0], \
                     [0, 0, 0.4, 0, 0, 0.6, 0, 0, 0, 0]])

    assert_str = 'test_placeholder_management failed'
    assert (np.array_equal(y.eval(feed_dict), y_gt)), assert_str

    assert_str = '__getitem__ method of PlaceholderManager class failed'
    assert ('word_embed' in plh_mgr['word_embed'].name), assert_str

    tf.reset_default_graph()


def test_var_collect():
    g = tf.Graph()
    with g.as_default():
        with tf.name_scope('scope1') as scope1:
            a = tf.Variable(tf.constant(1.0, shape=[1]), name='a')
            b = tf.Variable(tf.constant(2.0, shape=[1]), name='b')
        with tf.name_scope('scope2') as scope2:
            c = tf.Variable(tf.constant(3.0, shape=[1]), name='c')

    vars_g = var_collect.collect_all(graph=g)
    vars_g_scope1 = var_collect.collect_scope('scope1', graph=g)
    var_g_scope1_a = var_collect.collect_name('scope1/a', graph=g)
    var_g_dict = var_collect.collect_list(['scope1/a', 'scope2/c'], graph=g)

    # 'in' is used instead of '==' because variables have device id as well
    list_cmp = lambda var_name_list, var_obj_list: \
        all([v[0] in v[1].name for v in zip(var_name_list, var_obj_list)])

    all_vars = ['scope1/a', 'scope1/b', 'scope2/c']
    assert (list_cmp(all_vars, vars_g)), 'collect_all failed'
    assert (list_cmp(['scope1/a', 'scope1/b'], vars_g_scope1)), \
        'collect_scope failed'
    assert (list_cmp(['scope1/a'], [var_g_scope1_a])), 'collect_name failed'

    dict_check = lambda var_dict: \
        all([key in value.name for key, value in var_dict.items()])
    assert (dict_check(var_g_dict)), 'collect_list failed'

    # Default Graph
    d = tf.Variable(tf.constant(4.0, shape=[1]), name='d')
    e = tf.Variable(tf.constant(5.0, shape=[1]), name='e')

    def_graph_vars = var_collect.collect_all()
    assert (list_cmp(['d', 'e'], def_graph_vars)), \
        'collect_all failed on default graph'

    tf.reset_default_graph()


def test_var_collect_type():
    g = tf.Graph()
    with g.as_default():
        with tf.name_scope('scope1') as scope1:
            a = tf.Variable(
                tf.constant(1.0, shape=[1]),
                name='a', trainable=True)
            b = tf.Variable(
                tf.constant(1.0, shape=[1]),
                name='b',
                trainable=False)
            c = tf.Variable(
                tf.constant(1.0, shape=[1]),
                name='c',
                trainable=False)
        with tf.name_scope('scope2') as scope2:
            a = tf.Variable(
                tf.constant(1.0, shape=[1]),
                name='a',
                trainable=False)

    vars_all_1 = var_collect.collect_scope('scope1', graph=g)
    assert (len(vars_all_1) == 3)
    vars_trainable_1 = var_collect.collect_scope(
        'scope1', graph=g,
        var_type=tf.GraphKeys.TRAINABLE_VARIABLES)
    assert (len(vars_trainable_1) == 1)
    vars_all_2 = var_collect.collect_scope('scope2', graph=g)
    assert (len(vars_all_2) == 1)
    with pytest.raises(AssertionError):
        vars_trainable_2 = var_collect.collect_scope(
            'scope2',
            graph=g,
            var_type=tf.GraphKeys.TRAINABLE_VARIABLES)
