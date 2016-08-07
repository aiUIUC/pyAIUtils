""" This module defines a hepler class for multiple rate optimizers in Tensorflow

The MultiRateOptimizer class provides a slightly simpler version of the tf.train.Optimizer API
 - compute_gradients(loss)
 - apply_gradients(grads_and_vars, global_step)
 - minimize(loss, global_step)

To initialize:
 - MultiRateOptimizer([default_optimizer=tf.train.GradientDescentOptimizer])

To add variables and corresponding optimizers use add_variables as
 - add_variables(variables, optimizer)
 - add_variables(variables, learning_rate, [other_params])

Usage 1:
    a = tf.Variable(1)
    b = tf.Variable(2)
    c = tf.Variable(3)

    loss = f(a,b,c)

    optimizer = MultiRateOptimizer(tf.train.GradientDescentOptimizer)
    optimizer.add_variables([a,b], learning_rate=.1)
    optimizer.add_variables([c], learning_rate=.01)

    min_opt = optimizer.minimize(loss)

    min_opt.eval()

Usage 2:
    a = tf.Variable(1)
    b = tf.Variable(2)
    c = tf.Variable(3)

    loss = f(a,b,c)

    optimizer = MultiRateOptimizer()
    optimizer.add_variables([a,b], tf.train.GradientDescentOptimizer(.1))
    optimizer.add_variables([c], tf.train.GradientDescentOptimizer(.01))

    min_opt = optimizer.minimize(loss)

    min_opt.eval()
"""

import tensorflow as tf
import itertools


class MultiRateOptimizer():
    """ Class for managing a multi-rate optimization problem """

    def __init__(self, default_optimizer=None):
        self.optimizers = []
        self.variables = []
        self.default_optimizer = default_optimizer

    def check_variables(self, variables):
        """ Checks variables against the already added variables and returns a list
        of reused variables.

        variables (list of tf.variables): The variables to check

        return (list of tf.variables): The variables which are already known to this optimizer.
        """
        in_variables = set(itertools.chain(*self.variables))

        dupes = []
        for v in variables:
            if v in in_variables:
                dupes.append(v)

        return dupes

    def add_variables(self, variables, optimizer=None, learning_rate=None, other_params={}):
        """ Adds Variables and optimizers with different parameters.

        variables (list of tf.variables): the variables to optimize wrt.

        Either:
        optimizer (tf.train.Optimizer): The corresponding optimizer.
        Or:
        learning_rate (float): A learning rate to pass to the default_optimizer
        other_params (dict): A dictionary of param_name, value to pass the the default optimizer
        """

        print 'test'
        chck_vars = self.check_variables(variables)
        if len(chck_vars) != 0:
            raise ValueError('Expected all new variables, got overlap', *[v.name for v in chck_vars])
        assert(len(self.check_variables(variables)) == 0)
        self.variables.append(variables)

        if (optimizer is not None):
            self.optimizers.append(optimizer)
        else:
            if self.default_optimizer is None:
                raise ValueError('default_optimizer is None', 'When optimizer is not passed to add_variables, expect default_optimizer to be not None')

            self.optimizers.append(self.default_optimizer(learning_rate, **other_params))

        return self

    def compute_gradients(self, loss):
        """ Computes gradients of loss for the variables added to this object.

        This is the first part of minimize().  It returns a list of lists of
        (gradient, variable) pairs where "gradient" is the gradient for "variable".

        Args:
        - loss: A Tensor containing the value to minimize

        Returns:
        A list of (gradient, variable) pairs
        """

        all_vars = list(itertools.chain(*self.variables))
        gradients = tf.gradients(loss, all_vars)
        gradient_vars = zip(gradients, all_vars)
        shape_grad = []
        idx_grad = 0
        for vars in self.variables:
            shape_grad.append(gradient_vars[idx_grad:idx_grad + len(vars)])
            idx_grad += len(vars)
        return shape_grad

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """ Apply gradients to variables according to the optimizers.

        Args:
         - grads_and_vars: list of lists of (gradient, variable) pairs as returned by compute_gradients
         - global_step: Optional Variable to increment by one after the variables have been updated
         - name: Optional name for the returned operation.

        Returns:
        An operation that applies the specified gradients.  If global_step was not None, operation
        increments it also.
        """

        assert (len(self.optimizers) == len(grads_and_vars))

        apply_grad_list = []
        for optimizer, grad_and_var in zip(self.optimizers, grads_and_vars):
            apply_grad_list.append(optimizer.apply_gradients(grad_and_var))

        if global_step is not None:
            apply_grad_list.append(global_step.assign_add(1))

        return tf.group(*apply_grad_list, name=name)

    def minimize(self, loss, global_step=None, name=None):
        """
        Add operations to minimize loss by updating the variables added
        with add_variables.

        The method combines compute_gradients() and apply_gradients().
        """
        gradients = self.compute_gradients(loss)
        return self.apply_gradients(gradients, global_step, name=name)
