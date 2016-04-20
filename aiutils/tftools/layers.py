import numpy as np
import tensorflow as tf


def identity(input):
    return input


def full(input, out_dim, name, gain=np.sqrt(2), func=tf.nn.relu):
    """ Fully connected layer helper.
    
    Creates weights and bias parameters with good initial values. Then applies the matmul op and func.
    
    Args:
      input (tensor): Input to the layer. 
        Should have shape `[batch, in_dim]`.
        Must be one of the following types: `float32`, `float64`.
      out_dim (int): Number of output neurons.
      name (string): Name used by the `tf.variable_scope`.
      gain (float): Gain used when calculating stddev of weights.
        Suggest values: sqrt(2) for relu, 1.0 for identity.
      func (function): Function used to calculate neural activations.
      
    Returns:
      output (tensor): The neural activations for this layer.
        Will have shape `[batch, out_dim]`.
    """
    in_dim = input.get_shape().as_list()[-1]
    stddev = 1.0 * gain / np.sqrt(in_dim)
    with tf.variable_scope(name):
        w_init = tf.random_normal_initializer(stddev)
        b_init = tf.constant_initializer()
        w = tf.get_variable('w', shape=[in_dim, out_dim], initializer=w_init)
        b = tf.get_variable('b', shape=[out_dim], initializer=b_init)
        output = func(tf.matmul(input, w) + b)
    return output


def conv2d(input,
           filter_size,
           out_dim,
           name,
           strides=[1, 1, 1, 1],
           padding='SAME',
           gain=np.sqrt(2),
           func=tf.nn.relu):
    """ Conv2d layer helper.
    
    Creates filter and bias parameters with good initial values. Then applies the conv op and func.
    
    Args:
      input (tensor): Input to the layer. 
        Should have shape `[batch, in_height, in_width, in_dim]`.
        Must be one of the following types: `float32`, `float64`.
      filter_size (int): Width and height of square convolution filter.
      out_dim (int): Number of output filters.
      name (str): Name used by the `tf.variable_scope`.
      strides (List[int]): The stride of the sliding window for each dimension
        of `input`. Must be in the same order as the dimension specified with format.
      padding (str): A `string` from: `'SAME', 'VALID'`.
        The type of padding algorithm to use.
      gain (float): Gain used when calculating stddev of weights.
        Suggest values: sqrt(2) for relu, 1.0 for identity.
      func (function): Function used to calculate neural activations.
      
    Returns:
      output (tensor): The neural activations for this layer.
        Will have shape `[batch, in_weight, in_width, out_dim]`.
    """
    in_dim = input.get_shape().as_list()[-1]
    stddev = 1.0 * gain / np.sqrt(filter_size * filter_size * in_dim)
    with tf.variable_scope(name):
        w_init = tf.random_normal_initializer(stddev)
        b_init = tf.constant_initializer()
        w = tf.get_variable('w',
                            shape=[filter_size, filter_size, in_dim, out_dim],
                            initializer=w_init)
        b = tf.get_variable('b', shape=[out_dim], initializer=b_init)
        output = func(tf.nn.conv2d(input,
                                   w,
                                   strides=strides,
                                   padding=padding) + b)
    return output


def batch_norm(input, name):
    """ Batch norm layer helper.
    
    Gets mean and variance, and creates offset and scale parameters with good initial values.
    Then applies the batch_normalization op.
    
    Args:
      input (tensor): Input to the layer. 
        Should have shape `[batch, in_dim]` or `[batch, in_height, in_width, in_dim]`.
        Must be one of the following types: `float32`, `float64`.
      name (str): Name used by the `tf.variable_scope`.
      
    Returns:
      output (tensor): Batch normalized activations.
        Will have the same shape as input.
    """
    rank = len(input.get_shape().as_list())
    in_dim = input.get_shape().as_list()[-1]
    if rank == 2:
        axes = [0]
    elif rank == 4:
        axes = [0, 1, 2]
    else:
        raise ValueError('rank must be 2 or 4.')

    with tf.variable_scope(name):
        mean, variance = tf.nn.moments(input, axes)
        offset = tf.get_variable('offset',
                                 shape=[in_dim],
                                 initializer=tf.constant_initializer(0.0))
        scale = tf.get_variable('scale',
                                shape=[in_dim],
                                initializer=tf.constant_initializer(1.0))
        output = tf.nn.batch_normalization(input, mean, variance, offset,
                                           scale, 1e-5)
    return output
