import tensorflow as tf


def print_var_list(var_list, name='Variables'):
    print name + ': \n' + '[' + ', '.join([var.name for var in var_list]) + ']'


def collect_name(var_name, graph=None, var_type=tf.GraphKeys.VARIABLES):
    if graph == None:
        graph = tf.get_default_graph()

    var_list = graph.get_collection(var_type, scope=var_name)

    assert_str = "No variable exists with name '{}'".format(var_name)
    assert len(var_list) != 0, assert_str

    assert_str = \
        "Multiple variables exist with name_scope '{}'".format(var_name)
    assert len(var_list) == 1, assert_str

    return var_list[0]


def collect_scope(name_scope, graph=None, var_type=tf.GraphKeys.GLOBAL_VARIABLES):
    if graph == None:
        graph = tf.get_default_graph()

    var_list = graph.get_collection(var_type, scope=name_scope)

    assert_str = "No variable exists with name_scope '{}'".format(name_scope)
    assert len(var_list) != 0, assert_str

    return var_list


def collect_all(graph=None):
    if graph == None:
        graph = tf.get_default_graph()

    var_list = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    return var_list


def collect_all_trainable(graph=None):
    if graph == None:
        graph = tf.get_default_graph()

    var_list = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    return var_list


def collect_list(var_name_list, graph=None, var_type=tf.GraphKeys.GLOBAL_VARIABLES):
    var_dict = dict()
    for var_name in var_name_list:
        var_dict[var_name] = collect_name(
            var_name, graph=graph, var_type=var_type)

    return var_dict
