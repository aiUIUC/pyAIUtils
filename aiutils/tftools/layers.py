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

def conv2d_transpose(input,
           filter_size,
           out_dim,
           name,
           strides=[1, 1, 1, 1],
           gain=np.sqrt(2),
           func=tf.nn.relu,
           reuse_vars=False):
    padding = 'SAME'
    in_shape = input.get_shape().as_list()
    in_dim = in_shape[-1]
    out_shape = [x*y for x,y in zip(in_shape, strides)]
    out_shape[-1] = out_dim

    stddev = 1.0 * gain / np.sqrt(filter_size * filter_size * in_dim)
    with tf.variable_scope(name, reuse=reuse_vars):
        w_init = tf.random_normal_initializer(stddev=stddev)
        b_init = tf.constant_initializer()
        w = tf.get_variable('w',
                            shape=[filter_size, filter_size, out_dim, in_dim],
                            initializer=w_init)
        b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

        output = tf.nn.conv2d_transpose(input, w, out_shape, strides=strides, padding=padding) + b
        if func is not None:
            output = func(output)

    tf.add_to_collection('to_regularize', w)
    return output


def atrous_conv2d(input,
                  filter_size,
                  out_dim,
                  name,
                  rate=1,
                  padding='SAME',
                  gain=np.sqrt(2),
                  func=tf.nn.relu,
                  reuse_vars=False):
    """atrous_conv2d layer helper.

    Creates filter and bias parameters with good initial values.
    Then applies the atrous conv op and func.  See tf.nn.atrous_conv2d
    documentation for additional information.

    Args:
      input (tensor): Input to the layer.
        Should have shape `[batch, in_height, in_width, in_dim]`.
        Must be one of the following types: `float32`, `float64`.
      filter_size (int): Width and height of square convolution filter.
      out_dim (int): Number of output filters.
      name (str): Name used by the `tf.variable_scope`.
      rate (int): If greater than one, performs the conv operation with a filter
      with rate-1 0's between tow consecutive values
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

        output = tf.nn.atrous_conv2d(input, w, rate=rate, padding=padding) + b
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


def dropout(input, training=True, keep_prob=.8, noise_shape=None, seed=None):
    """Adds a dropout layer, which is optionally active for ease of
    constructing training and test graphs which use the same code.

    Args:
        input (tensor): Tensor to droupout.
        training (bool or bool tensor): Determines whether the dropout op is active
        keep_prob (float): Deterimnes how much of the vector to not dropout
        noise_shape (one-d int32 tensor): If noise_shape is specified, it must be broadcastable
            to the shape of input. Dimensions with noise_shape[i] == shape(input)[i]
            make independent decissions.
        seed (int): Used to create random seeds. See tf.set_random_seed for behavior.


    Returns:
        (tensor): input tensor, or dropped out tensor depending on training.
    """

    if type(training) is type(True):
        if training:
            return tf.nn.dropout(input,
                                 keep_prob,
                                 noise_shape=noise_shape,
                                 seed=seed)
        else:
            return input
    else:
        return tf.cond(training,
                lambda: tf.nn.dropout(input, keep_prob, noise_shape=noise_shape, seed=seed),
                lambda: input)
