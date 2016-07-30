""" This module defines a hepler class for multiple rate optimizers in Tensorflow

The MultiRateOptimizer class provides a slightly simpler version of the tf.train.Optimizer API
 - compute_gradients(loss)
 - apply_gradients(grads_and_vars, global_step)
 - minimize(loss, global_step)

To add variables and corresponding optimizers use
 - add_variables(variables, optimizer)

Usage:
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

    def __init__(self):
        self.optimizers = []
        self.variables = []

    def add_variables(self, variables, optimizer):
        """ Adds Variables and optimizers with different parameters.

        variables (list of tf.variables): the variables to optimize wrt.
        optimizer (tf.train.Optimizer): The corresponding optimizer.

        """

        self.variables.append(variables)
        self.optimizers.append(optimizer)

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
