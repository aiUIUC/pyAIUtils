import numpy as np
import tensorflow as tf


def identity(input):
    return input


def full(input, out_dim, name, gain=np.sqrt(2), func=tf.nn.relu):
    """ Fully connected layer helper.
    
    Creates weights and bias parameters with good initial values. Then applies the matmul op and func.
    
    Args:
      input: A `Tensor` of shape `[batch, in_dim]`.
        Must be one of the following types: `float32`, `float64`.
      out_dim: An `int`. Number of output neurons.
      name: A `string`. Name used by the `tf.variable_scope`.
      gain: A `float`. Suggest values: sqrt(2) for relu, 1.0 for identity.
      func: A `function`. 
      
    Returns:
      output: A `Tensor` of shape `[batch, in_weight, in_width, out_dim]`.
    """
    in_dim = input.get_shape().as_list()[-1]
    stddev = 1.0 * gain / np.sqrt(in_dim)
    with tf.variable_scope(name):
        w_init = tf.random_normal_initializer(stddev)
        b_init = tf.constant_initializer()
        w = tf.get_variable('w',
                            shape=[in_dim, out_dim],
                            initializer=w_init)
        b = tf.get_variable('b',
                            shape=[out_dim],
                            initializer=b_init)
        output = func(tf.matmul(input, w) + b)
    return output

def conv2d(input, filter_size, out_dim, name, strides=[1, 1, 1, 1], padding='SAME', gain=np.sqrt(2), func=tf.nn.relu):
    """ Conv2d layer helper.
    
    Creates filter and bias parameters with good initial values. Then applies the conv op and func.
    
    Args:
      input: A `Tensor` of shape `[batch, in_height, in_width, in_dim]`.
        Must be one of the following types: `float32`, `float64`.
      filter_size: An `int`. Width and height of square kernel.
      out_dim: An `int`. Number of output filters.
      name: A `string`. Name used by the `tf.variable_scope`.
      strides: A list of `ints`.
        1-D of length 4.  The stride of the sliding window for each dimension
        of `input`. Must be in the same order as the dimension specified with format.
      padding: A `string` from: `"SAME", "VALID"`.
        The type of padding algorithm to use.
      gain: A `float`. Suggest values: sqrt(2) for relu, 1.0 for identity.
      func: A `function`. 
      
    Returns:
      output: A `Tensor` of shape `[batch, in_weight, in_width, out_dim]`.
    """
    in_dim = input.get_shape().as_list()[-1]
    stddev = 1.0 * gain / np.sqrt(filter_size*filter_size*in_dim)
    with tf.variable_scope(name):
        w_init = tf.random_normal_initializer(stddev)
        b_init = tf.constant_initializer()
        w = tf.get_variable('w',
                            shape=[filter_size, filter_size, in_dim, out_dim],
                            initializer=w_init)
        b = tf.get_variable('b',
                            shape=[out_dim],
                            initializer=b_init)
        output = func(tf.nn.conv2d(input, w, strides=strides, padding=padding) + b)
    return output