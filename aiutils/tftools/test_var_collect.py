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

    # Default Graph
    d = tf.Variable(tf.constant(4.0, shape=[1]), name='d')
    e = tf.Variable(tf.constant(5.0, shape=[1]), name='e')

    def_graph_vars = var_collect.collect_all()
    assert (list_cmp(['d', 'e'], def_graph_vars)), \
        'collect_all failed on default graph'


if __name__ == '__main__':
    #    var_collection_example()
    test_var_collect()
