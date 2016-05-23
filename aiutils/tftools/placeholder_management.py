"""This module defines a helper class for managing placeholders in Tensorflow.

The PlholderManager class allows easy management of placeholders including 
adding placeholders to a Tensorflow graph, producing easy access to the added
placeholders using a dictionary with placeholder names as keys, create feed 
dictionary from a given input dictionary, and print placeholders and feed dict 
to standard output. 

More importantly, the class allows sparse scipy matrices to 
be passed into graphs (Tensorflow currently allows only dense matrices to be fed
into placeholders).

Usage:
    pm = PlaceholderManager()
    pm.add_placeholder('x1', tf.float64, [1,2])
    pm.add_placeholder('x2', tf.float64, [1,2])

    # Use placeholders in your graph
    y = pm['x1'] + pm['x2']
   
    # Create feed dictionary
    feed_dict = pm.get_feed_dict({'x1': np.array([3.0, 4.0]), 
                                  'x2': np.array([5.0, 2.0])})
    y.eval(feed_dict)
"""
import tensorflow as tf
import numpy as np
import scipy.sparse as sps
import pdb


class PlaceholderManager():
    """Class for managing placeholders."""

    def __init__(self):
        self._placeholders = dict()
        self.issparse = dict()

    def add_placeholder(self, name, dtype, shape=None, sparse=False):
        """Add placeholder.
        
        If the sparse is True then 3 placeholders are automatically created 
        corresponding to the indices and values of the non-zero entries, and 
        shape of the sparse matrix. The user does not need to keep track of 
        these and can directly pass a sparse scipy matrix as input and a 
        Tensorflow SparseTensor object is made available for use in the graph.

        Args:
            name (str): Name of the placeholder. 
            dtype (tf.Dtype): Data type for the placeholder.
            shape (list of ints): Shape of the placeholder.
            sparse (bool): Specifies if the placeholder takes sparse inputs.
        """

        self.issparse[name] = sparse
        if not sparse:
            self._placeholders[name] = tf.placeholder(dtype, shape, name)

        else:
            name_indices = name + '_indices'
            name_values = name + '_values'
            name_shape = name + '_shape'

            self._placeholders[name_indices] = tf.placeholder(tf.int64, [None, 2],
                                                           name_indices)
            self._placeholders[name_values] = tf.placeholder(dtype, [None],
                                                          name_values)
            self._placeholders[name_shape] = tf.placeholder(tf.int64, [2],
                                                         name_shape)

    def __getitem__(self, name):
        """Returns placeholder with the given name.
        
        Usage:
            plh_mgr = PlaceholderManager()
            plh_mgr.add_placeholder('var_name', tf.int64, sparse=True)
            placeholder = plh_mgr['var_name']
        """
        sparse = self.issparse[name]
        if not sparse:
            placeholder = self._placeholders[name]
        else:
            placeholder_indices = self._placeholders[name + '_indices']
            placeholder_values = self._placeholders[name + '_values']
            placeholder_shape = self._placeholders[name + '_shape']
            sparse_tensor = tf.SparseTensor(
                placeholder_indices, placeholder_values, placeholder_shape)
            placeholder = sparse_tensor

        return placeholder

    def get_placeholders(self):
        """Returns a dictionary of placeholders with names as keys.
        
        The returned dictionary provides an easy way of refering to the 
        placeholders and passing them to graph construction or evaluation 
        functions.
        """
        placeholders = dict()
        for name in self.issparse.keys():
            placeholders[name] = self[name]

        return placeholders

    def get_feed_dict(self, inputs):
        """Returns a feed dictionary that can be passed to eval() or run().

        This method creates a feed dictionary from provided inputs that can be
        passed directly into eval() or run() routines in Tensorflow. 

        Usage:
            pm = PlaceholderManager()
            pm.add_placeholder('x', tf.float64, [1,2])
            pm.add_placeholder('y', tf.float64, [1,2])
            z = pm['x'] + pm['y']
            inputs = {
                'x': np.array([3.0, 4.0]), 
                'y': np.array([5.0, 2.0])
            }
            feed_dict = pm.get_feed_dict(inputs)
            z.eval(feed_dict)

        Args:
            inputs (dict): A dictionary with placeholder names as keys and the 
                inputs to be passed in as the values. For 'sparse' placeholders
                only the sparse scipy matrix needs to be passed in instead of 
                3 separate dense matrices of indices, values and shape.
        """
        feed_dict = dict()
        for name, input_value in inputs.items():
            try:
                placeholder_sparsity = self.issparse[name]
                input_sparsity = sps.issparse(input_value)
                assert_str = 'Sparsity of placeholder and input do not match'
                assert (placeholder_sparsity == input_sparsity), assert_str
                if not input_sparsity:
                    placeholder = self._placeholders[name]
                    feed_dict[placeholder] = input_value

                else:
                    I, J, V = sps.find(input_value)

                    placeholder_indices = self._placeholders[name + '_indices']
                    placeholder_values = self._placeholders[name + '_values']
                    placeholder_shape = self._placeholders[name + '_shape']

                    feed_dict[placeholder_indices] = \
                        np.column_stack([I, J]).astype(np.int64)
                    feed_dict[placeholder_shape] = \
                        np.array(input_value.shape).astype(np.int64)
                    if placeholder_values.dtype == tf.int64:
                        feed_dict[placeholder_values] = V.astype(np.int64)
                    else:
                        feed_dict[placeholder_values] = V

            except KeyError:
                print "No placeholder with name '{}'".format(name)
                raise

        return feed_dict

    def feed_dict_debug_string(self, feed_dict):
        """Returns feed dictionary as a neat string.

        Args:
            feed_dict (dict): Output of get_feed_dict() or a dictionary with 
                placeholders as keys and the values to be fed into the graph as 
                values. 
        """

        debug_str = 'feed_dict={\n'
        for plh, value in feed_dict.items():
            debug_str += '{}: \n{}\n'.format(plh, value)
        debug_str += '}'
        return debug_str

    def placeholder_debug_string(self, placeholders=None):
        """Returns placeholder information as a neat string.
        
        Args:
            placeholders (dict): Output of get_placeholders() or None in which
                case self._placeholders is used.
        """

        if not placeholders:
            placeholders = self._placeholders

        debug_str = 'placeholders={\n'
        for name, plh in placeholders.items():
            debug_str += "    '{}': {}\n".format(name, plh)
        debug_str += '}'
        return debug_str

    def __str__(self):
        """Allows PlaceholderManager object to be used with print or str()."""
        return self.placeholder_debug_string()


