import tensorflow as tf
import pdb

import aiutils.tftools.var_collect as var_collect


def var_collection_example():
    g1 = tf.Graph()
    with g1.as_default():
        with tf.name_scope('scope1') as scope1:
            a = tf.Variable(tf.constant(1.0, shape=[1]), name='a')
            b = tf.Variable(tf.constant(2.0, shape=[1]), name='b')
        with tf.name_scope('scope2') as scope2:
            c = tf.Variable(tf.constant(3.0, shape=[1]), name='c')

    g2 = tf.Graph()
    with g2.as_default():
        with tf.name_scope('scope1') as scope1:
            a = tf.Variable(tf.constant(4.0, shape=[1]), name='a')
            b = tf.Variable(tf.constant(5.0, shape=[1]), name='b')
        with tf.name_scope('scope2') as scope2:
            c = tf.Variable(tf.constant(6.0, shape=[1]), name='c')

    vars_g1 = var_collect.collect_all(graph=g1)
    vars_g1_scope1 = var_collect.collect_scope('scope1', graph=g1)
    var_g1_scope1_a = var_collect.collect_name('scope1/a', graph=g1)

    vars_g2 = var_collect.collect_all(graph=g2)
    vars_g2_dict = var_collect.collect_list(
        ['scope1/a', 'scope1/b', 'scope2/c'],
        graph=g2)

    sess = tf.Session(graph=g1)
    sess.run(tf.initialize_variables(vars_g1))
    y_hat = [var.eval(sess)[0] for var in vars_g1]
    y = [1.0, 2.0, 3.0]
    print 'Graph g1: '
    print 'y: [' + ', '.join([str(l) for l in y]) + ']'
    print 'y_hat: [' + ', '.join([str(l) for l in y_hat]) + ']'
    sess.close()

    sess = tf.Session(graph=g2)
    sess.run(tf.initialize_variables(vars_g2))
    y_hat = [var.eval(sess)[0] for var in vars_g2]
    y = [4.0, 5.0, 6.0]
    print 'Graph g2: '
    print 'y: [' + ', '.join([str(l) for l in y]) + ']'
    print 'y_hat: [' + ', '.join([str(l) for l in y_hat]) + ']'

    var_collect.print_var_list(vars_g1, name='vars_g1')
    var_collect.print_var_list(vars_g2, name='vars_g2')
    var_collect.print_var_list(vars_g1_scope1, name='vars_g1_scope1')
    var_collect.print_var_list([var_g1_scope1_a], name='vars_g1_scope1_a')

    print 'vars_g2_dict = {'
    for key, value in vars_g2_dict.items():
        print '    {}: {},'.format(key, value.eval(sess)[0])
    print '}'
    sess.close()


if __name__ == '__main__':
    var_collection_example()
