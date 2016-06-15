import numpy as np
import tensorflow as tf
from batch_normalizer import BatchNorm


def full(input,
         out_dim,
         name,
         gain=np.sqrt(2),
         func=tf.nn.relu,
         reuse_vars=False):
    """ Fully connected layer helper.

    Creates weights and bias parameters with good initial values. 
    Then applies the matmul op and func.

    Args:
      input (tensor): Input to the layer.
        Should have shape `[batch, in_dim]`.
        Must be one of the following types: `float32`, `float64`.
      out_dim (int): Number of output neurons.
      name (string): Name used by the `tf.variable_scope`.
      gain (float): Gain used when calculating stddev of weights.
        Suggest values: sqrt(2) for relu, 1.0 for identity.
      func (function): Function used to calculate neural activations.
        If `None` uses identity.
      reuse_vars (bool): Determine whether the layer should reuse variables
        or construct new ones.  Equivalent to setting reuse in variable_scope

    Returns:
      output (tensor): The neural activations for this layer.
        Will have shape `[batch, out_dim]`.
    """
    in_dim = input.get_shape().as_list()[-1]
    stddev = 1.0 * gain / np.sqrt(in_dim)
    with tf.variable_scope(name, reuse=reuse_vars):
        w_init = tf.random_normal_initializer(stddev=stddev)
        b_init = tf.constant_initializer()
        w = tf.get_variable('w', shape=[in_dim, out_dim], initializer=w_init)
        b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

        output = tf.matmul(input, w) + b
        if func is not None:
            output = func(output)

    tf.add_to_collection('to_regularize', w)
    return output


def conv2d(input,
           filter_size,
           out_dim,
           name,
           strides=[1, 1, 1, 1],
           padding='SAME',
           gain=np.sqrt(2),
           func=tf.nn.relu,
           reuse_vars=False):
    """ Conv2d layer helper.

    Creates filter and bias parameters with good initial values. 
    Then applies the conv op and func.

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
        If `None` uses identity.
      reuse_vars (bool): Determine whether the layer should reuse variables
        or construct new ones.  Equivalent to setting reuse in variable_scope

    Returns:
      output (tensor): The neural activations for this layer.
        Will have shape `[batch, in_weight, in_width, out_dim]`.
    """
    in_dim = input.get_shape().as_list()[-1]
    stddev = 1.0 * gain / np.sqrt(filter_size * filter_size * in_dim)
    with tf.variable_scope(name, reuse=reuse_vars):
        w_init = tf.random_normal_initializer(stddev=stddev)
        b_init = tf.constant_initializer()
        w = tf.get_variable('w',
                            shape=[filter_size, filter_size, in_dim, out_dim],
                            initializer=w_init)
        b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

        output = tf.nn.conv2d(input, w, strides=strides, padding=padding) + b
        if func is not None:
            output = func(output)

    tf.add_to_collection('to_regularize', w)
    return output


def batch_norm(input,
               training=tf.constant(True),
               decay=0.95,
               epsilon=1e-4,
               name='bn',
               reuse_vars=False):
    """Adds a batch normalization layer.

    Args:
        input (tensor): Tensor to be batch normalized
        training (bool tensor): Boolean tensor of shape []
        decay (float): Decay used for exponential moving average
        epsilon (float): Small constant added to variance to prevent
            division of the form 0/0
        name (string): variable scope name
        reuse_vars (bool): Value passed to reuse keyword argument of 
            tf.variable_scope. This only reuses the offset and scale variables, 
            not the moving average shadow variable.

    Returns:
        output (tensor): Batch normalized output tensor
    """
    bn = BatchNorm(input, training, decay, epsilon, name, reuse_vars)
    output = bn.output

    return output
