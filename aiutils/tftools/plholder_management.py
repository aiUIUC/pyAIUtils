"""This module defines a helper class for managing placeholders in Tensorflow.

The PlholderManager class allows easy management of placeholders including 
adding placeholders to a Tensorflow graph, producing easy access to the added
placeholders using a dictionary with placeholder names as keys, create feed 
dictionary from a given input dictionary, and print placeholders and feed dict 
to standard output. 

More importantly, the class allows sparse scipy matrices to 
be passed into graphs (Tensorflow currently allows only dense matrices to be fed
into placeholders).
"""
import tensorflow as tf
import numpy as np
import scipy.sparse as sps
import pdb


class PlholderManager():
    """Class for managing placeholders."""

    def __init__(self):
        self._plholders = dict()
        self.issparse = dict()

    def add_plholder(self, name, dtype, shape=None, sparse=False):
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
            self._plholders[name] = tf.placeholder(dtype, shape, name)

        else:
            name_indices = name + '_indices'
            name_values = name + '_values'
            name_shape = name + '_shape'

            self._plholders[name_indices] = tf.placeholder(tf.int64, [None, 2],
                                                           name_indices)
            self._plholders[name_values] = tf.placeholder(dtype, [None],
                                                          name_values)
            self._plholders[name_shape] = tf.placeholder(tf.int64, [2],
                                                         name_shape)

    def get_plholders(self):
        """Returns a dictionary of placeholders with names as keys.
        
        The returned dictionary provides an easy way of refering to the 
        placeholders and passing them to graph construction or evaluation 
        functions.
        """
        plholders = dict()
        for name, sparse in self.issparse.items():
            if not sparse:
                plholders[name] = self._plholders[name]
            else:
                plholder_indices = self._plholders[name + '_indices']
                plholder_values = self._plholders[name + '_values']
                plholder_shape = self._plholders[name + '_shape']
                sparse_tensor = tf.SparseTensor(
                    plholder_indices, plholder_values, plholder_shape)
                plholders[name] = sparse_tensor

        return plholders

    def __getitem__(self, name):
        """Returns placeholder with the given name.
        
        Usage:
            plh_mgr = PlholderManager()
            plh_mgr.add_plholder('var_name', tf.int64, sparse=True)
            plholder = plh_mgr['var_name']
        """
        sparse = self.issparse[name]
        if not sparse:
            plholder = self._plholders[name]
        else:
            plholder_indices = self._plholders[name + '_indices']
            plholder_values = self._plholders[name + '_values']
            plholder_shape = self._plholders[name + '_shape']
            sparse_tensor = tf.SparseTensor(
                plholder_indices, plholder_values, plholder_shape)
            plholder[name] = sparse_tensor

        return plholder

    def get_feed_dict(self, inputs):
        """Returns a feed dictionary that can be passed to eval() or run().

        This method creates a feed dictionary from provided inputs that can be
        passed directly into eval() or run() routines in Tensorflow. 

        Args:
            inputs (dict): A dictionary with placeholder names as keys and the 
                inputs to be passed in as the values. For 'sparse' placeholders
                only the sparse scipy matrix needs to be passed in instead of 
                3 separate dense matrices of indices, values and shape.
        """
        feed_dict = dict()
        for name, input_value in inputs.items():
            try:
                plholder_sparsity = self.issparse[name]
                input_sparsity = sps.issparse(input_value)
                assert_str = 'Sparsity of placeholder and input do not match'
                assert (plholder_sparsity == input_sparsity), assert_str
                if not input_sparsity:
                    plholder = self._plholders[name]
                    feed_dict[plholder] = input_value

                else:
                    I, J, V = sps.find(input_value)

                    plholder_indices = self._plholders[name + '_indices']
                    plholder_values = self._plholders[name + '_values']
                    plholder_shape = self._plholders[name + '_shape']

                    feed_dict[plholder_indices] = \
                        np.column_stack([I, J]).astype(np.int64)
                    feed_dict[plholder_shape] = \
                        np.array(input_value.shape).astype(np.int64)
                    if plholder_values.dtype == tf.int64:
                        feed_dict[plholder_values] = V.astype(np.int64)
                    else:
                        feed_dict[plholder_values] = V

            except KeyError:
                print "No placeholder with name '{}'".format(name)
                raise

        return feed_dict

    def print_feed_dict(self, feed_dict):
        """Prints feed dictionary to standard output neatly."""

        print 'feed_dict={\n'
        for plh, value in feed_dict.items():
            print '{}: \n{}\n'.format(plh, value)
        print '}'

    def print_plholders(self, plholders=None):
        """Prints placeholders to standard output neatly."""

        if not plholders:
            plholders = self._plholders

        print 'placeholders={'
        for name, plh in plholders.items():
            print "    '{}': {}".format(name, plh)
        print '}'
