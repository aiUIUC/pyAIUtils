import tensorflow as tf
import pdb

import var_collect


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

    g2 = tf.Graph()
    with g2.as_default():
        # Default Graph
        d = tf.Variable(tf.constant(4.0, shape=[1]), name='d')
        e = tf.Variable(tf.constant(5.0, shape=[1]), name='e')

        def_graph_vars = var_collect.collect_all()
        assert (list_cmp(['d', 'e'], def_graph_vars)), \
            'collect_all failed on default graph'


def test_var_collect_type():
    g = tf.Graph()
    with g.as_default():
        with tf.name_scope('scope1') as scope1:
            a = tf.Variable(tf.constant(1.0, shape=[1]), name='a', trainable=True)
            b = tf.Variable(tf.constant(1.0, shape=[1]), name='b', trainable=False)
            c = tf.Variable(tf.constant(1.0, shape=[1]), name='c', trainable=False)
        with tf.name_scope('scope2') as scope2:
            a = tf.Variable(tf.constant(1.0, shape=[1]), name='a', trainable=False)

    vars_all_1 = var_collect.collect_scope('scope1', graph=g)
    assert(len(vars_all_1) == 3)
    vars_trainable_1 = var_collect.collect_scope('scope1', graph=g, var_type=tf.GraphKeys.TRAINABLE_VARIABLES)
    assertEqual(len(vars_trainable_1), 1)
    vars_all_2 = var_collect.collect_scope('scope2', graph=g)
    assert(len(vars_all_1) == 1)
    vars_trainable_2 = var_collect.collect_scope('scope2', graph=g, var_type=tf.GraphKeys.TRAINABLE_VARIABLES)
    assert(len(vars_trainable_1) == 0)

if __name__ == '__main__':
    #    var_collection_example()
    test_var_collect()
