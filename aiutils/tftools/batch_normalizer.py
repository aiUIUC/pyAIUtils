"""A helper class for managing batch normalization state.

This class is designed to simplify adding batch normalization
(http://arxiv.org/pdf/1502.03167v3.pdf) to your model by
managing the state variables associated with it.

Important use note:  The function get_assigner() returns
an op that must be executed to save the updated state.
A suggested way to do this is to make execution of the
model optimizer force it, e.g., by:

  update_assignments = tf.group(bn1.get_assigner(),
                                bn2.get_assigner())
  with tf.control_dependencies([optimizer]):
    optimizer = tf.group(update_assignments)
"""

import pdb
import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops


class BatchNormalizer(object):
    """Helper class that groups the normalization logic and variables.

    Use:
        ewma = tf.train.ExponentialMovingAverage(decay=0.99)
        bn = BatchNormalizer(input, ewma, 'name',  0.001)
        update_assignments = bn.get_assigner()
        x = bn.normalize(y, train=training)
        (the output x will be batch-normalized).
    """

    def __init__(self, input, ewma_trainer, name, epsilon=1e-5):
        rank = len(input.get_shape())
        in_dim = input.get_shape().as_list()[-1]

        if rank == 2:
            self.axes = [0]
        elif rank == 4:
            self.axes = [0, 1, 2]
        else:
            raise ValueError('Input tensor must have rank 2 or 4.')

        self.mean = tf.Variable(
            tf.constant(0.0,
                        shape=[in_dim]),
            trainable=False)
        self.variance = tf.Variable(
            tf.constant(1.0,
                        shape=[in_dim]),
            trainable=False)
        with tf.variable_scope(name):
            self.offset = tf.get_variable(
                'offset',
                shape=[in_dim],
                initializer=tf.constant_initializer(0.0))
            self.scale = tf.get_variable(
                'scale',
                shape=[in_dim],
                initializer=tf.constant_initializer(1.0))
        self.ewma_trainer = ewma_trainer
        self.epsilon = epsilon

        # initializes the moving average
        self.assigner = None
        self.assigner = self.get_assigner()

    def get_assigner(self):
        """Returns an EWMA apply op that must be invoked after optimization."""
        if not self.assigner:
            self.assigner = self.ewma_trainer.apply([self.mean, self.variance])
        return self.assigner

    def normalize(self, input, train=True):
        """Returns a batch-normalized version of x."""
        if train:
            mean, variance = tf.nn.moments(input, self.axes)
            assign_mean = self.mean.assign(mean)
            assign_variance = self.variance.assign(variance)
            with tf.control_dependencies([assign_mean, assign_variance]):
                return tf.nn.batch_normalization(input, mean, variance,
                                                 self.offset, self.scale,
                                                 self.epsilon)
        else:
            mean = self.ewma_trainer.average(self.mean)
            variance = self.ewma_trainer.average(self.variance)
            local_offset = tf.identity(self.offset)
            local_scale = tf.identity(self.scale)
            return tf.nn.batch_normalization(
                input, mean, variance, local_offset, local_scale, self.epsilon)


def batch_normalize(input, phase_train, name):
    ema = tf.train.ExponentialMovingAverage(.99)
    bn = BatchNormalizer(input, ema, name)
    tf.add_to_collection('batch_norm_average_update', bn.get_assigner())
    return control_flow_ops.cond(phase_train,
                                 lambda: bn.normalize(input, True),
                                 lambda: bn.normalize(input, False))


class BatchNorm():
    def __init__(self, input, training, decay=0.95, epsilon=1e-4, name='bn'):
        self.decay = decay
        self.epsilon = epsilon
        self.batchnorm(input, training, name)

    def batchnorm(self, input, training, name):
        with tf.name_scope(name) as bn:
            rank = len(input.get_shape().as_list())
            in_dim = input.get_shape().as_list()[-1]

            if rank == 2:
                print('rank2')
                axes = [0]
            elif rank == 4:
                print('rank4')
                axes = [0, 1, 2]
            else:
                raise ValueError('Input tensor must have rank 2 or 4.')

            self.mean, self.variance = tf.nn.moments(input, axes)
            self.offset = tf.Variable(initial_value=tf.constant(value=0.0,
                                                                shape=[in_dim]),
                                      name='offset')
            self.scale = tf.Variable(initial_value=tf.constant(value=1.0,
                                                               shape=[in_dim]),
                                     name='scale')

            self.ema = tf.train.ExponentialMovingAverage(decay=self.decay)

            self.output = tf.cond(training,
                    lambda: self.get_normalizer(input, True),
                    lambda: self.get_normalizer(input, False))

    def get_normalizer(self, input, train_flag):
        if train_flag:
            ema_apply_op = self.ema.apply([self.mean, self.variance])
            with tf.control_dependencies([ema_apply_op]):
                self.output_training = tf.nn.batch_normalization(
                        input, self.mean, self.variance, self.offset,
                        self.scale, self.epsilon, 'normalizer_train'),
            return self.output_training
        else:
            self.output_test = tf.nn.batch_normalization(
                input, self.ema.average(self.mean), self.ema.average(self.variance), self.offset,
                self.scale, self.epsilon, 'normalizer_test')
            return self.output_test
    # def get_normalizer(self, input, train_flag):
    #     if train_flag:
    #         print 'train'
    #         def f1(): 
    #             with tf.control_dependencies([self.ema_apply_op]):
    #                 something = tf.nn.batch_normalization(
    #                     input, self.mean, self.variance, self.offset,
    #                     self.scale, self.epsilon, 'normalizer_train')
    #                 return something
    #         return f1
    #     else:
    #         print 'test'
    #         def f2():
    #             self.ema_apply_op = ema.apply([self.mean, self.variance])
    #         self.ema_mean = ema.average(self.mean)
    #         self.ema_var = ema.average(self.variance)
    #         return tf.nn.batch_normalization(
    #                 input, self.ema_mean, self.ema_var, self.offset,
    #                 self.scale, self.epsilon, 'normalizer_test')
    #         return f2

    def get_batch_moments(self):
        return self.mean, self.variance

    def get_ema_moments(self):
        return self.ema.average(self.mean), self.ema.average(self.variance)

    def get_offset_scale(self):
        return self.offset, self.scale


def batchnorm_failing_case():
    batch = 5
    width = 2
    height = 3
    channels = 4

    input_shape = [batch, width, height, channels]

    g = tf.Graph()
    with g.as_default():
        training = tf.placeholder(tf.bool, [])
        x = tf.placeholder(tf.float32, input_shape)
        bn = BatchNorm(x, training, name='bn')
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
        tf.reset_default_graph()

    sess.close()


    assert_str = 'batch mean and var are not used correctly' + \
                 'during training with batch norm'
    assert (np.all(y_eval1 == np.zeros(input_shape))), assert_str
    assert_str = 'batch mean and var are not used correctly' + \
                 'during training with batch norm'
    assert (np.all(y_eval2 == np.zeros(input_shape))), assert_str
    assert_str = 'ema mean is not updated during training with batch norm'
    assert (not np.all(ema_mean_eval1 == ema_mean_eval2)), assert_str


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
        bn = BatchNorm(x, training, name='bn')
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
        bn = BatchNorm(x, training, name='bn')
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

        print y.name

    sess.close()

    assert_str = 'ema mean and var are not used correctly' + \
                 'during testing with batch norm'
    assert (not np.all(y_eval1 == np.zeros(input_shape))), assert_str
    assert_str = 'ema mean and var are not used correctly' + \
                 'during testing with batch norm'
    assert (not np.all(y_eval2 == np.zeros(input_shape))), assert_str
    assert_str = 'ema mean is updated during testing with batch norm'
    assert (np.all(ema_mean_eval1 == ema_mean_eval2)), assert_str

if __name__=='__main__':
    batchnorm_failing_case()
    test_batchnorm_train_mode()
    test_batchnorm_test_mode()
