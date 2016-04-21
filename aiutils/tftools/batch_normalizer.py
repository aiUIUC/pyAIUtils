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

import tensorflow as tf
from tensorflow.python import control_flow_ops

class BatchNormalizer(object):
    """Helper class that groups the normalization logic and variables.

    Use:
        ewma = tf.train.ExponentialMovingAverage(decay=0.99)
        bn = BatchNormalizer(input, 0.001, ewma, True)
        update_assignments = bn.get_assigner()
        x = bn.normalize(y, train=training?)
        (the output x will be batch-normalized).
    """

    def __init__(self, input, epsilon, ewma_trainer, name):
        rank = len(input.get_shape().as_list())
        in_dim = input.get_shape().as_list()[-1]

        if rank == 2:
            self.axes = [0]
        elif rank == 4:
            self.axes = [0, 1, 2]
        else:
            raise ValueError('Input tensor must have rank 2 or 4.')

        self.mean = tf.Variable(tf.constant(0.0, shape=[in_dim]),
                trainable=False)
        self.variance = tf.Variable(tf.constant(1.0, shape=[in_dim]),
                trainable=False)
        with tf.variable_scope(name):
            self.beta = tf.get_variable('offset',
                    shape = [in_dim],
                    initializer = tf.constant_initializer(0.0))
            self.gamma = tf.get_variable('scale',
                    shape = [in_dim],
                    initializer=tf.constant_initializer(1.0))
        self.ewma_trainer = ewma_trainer
        self.epsilon = epsilon

        # initializes the moving average
        self.assigner = None
        self.assigner = self.get_assigner()

    def get_vars(self):
        return [self.beta, self.gamma]

    def get_weight_decay(self):
        return tf.nn.l2_loss(self.beta) + tf.nn.l2_loss(self.gamma)

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
                return tf.nn.batch_normalization(input, mean, variance, self.beta,
                        self.gamma, self.epsilon)
        else:
            mean = self.ewma_trainer.average(self.mean)
            variance = self.ewma_trainer.average(self.variance)
            local_beta = tf.identity(self.beta)
            local_gamma = tf.identity(self.gamma)
            return tf.nn.batch_normalization(input, mean, variance, local_beta,
                    local_gamma, self.epsilon)

def batch_normalize(input, phase_train, bn_obj):
    return control_flow_ops.cond(phase_train,
            lambda: bn_obj.normalize(input, True),
            lambda: bn_obj.normalize(input, False))

