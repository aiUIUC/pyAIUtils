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
import pprint as pp
import pdb


class PlaceholderManager():
    """Class for managing placeholders."""

    def __init__(self):
        self._placeholders = dict()
        self.issparse = set()
        self.islist = set()

    def add_placeholder(
            self,
            name,
            dtype,
            shape=None,
            list_len=None,
            sparse=False):
        """Add placeholder.
        Args:
            name (str): Name of the placeholder.
            dtype (tf.Dtype): Data type for the placeholder.
            shape (list of ints): Shape of the placeholder.
            list_len (int) : If not None a list of placeholders is created with
                list_len being the length of the list. plh[name] returns the 
                list of placeholders in this case.
            sparse (bool): If True, 3 placeholders are automatically created
                for passing in indices and values of the non-zero entries, and
                shape of the sparse matrix. In addition it automatically adds a
                tf.SparseTensor op which converts the 3 placeholders into a 
                single sparse tensor. plh[name] returns a dict with 'indices',
                'values', 'shape', and 'tensor' as keys. The first three may be
                used to pass in inputs as part of a feed dict, and the last can 
                be used for passing the sparse tensor to upstream ops. Instead 
                of keeping track of 3 tensors in feed dict, a sparse matrix can 
                be directly passed as input while using get_feed_dict method. 
                We will refer to the dict returned by plh[name] as a sparse 
                placeholder. 
        """
        if list_len:
            assert_string = 'list_len needs to be a positive integer'
            assert (isinstance(list_len, int) and list_len > 0), assert_string
            self.islist.add(name)
            self._placeholders[name] = [None] * list_len
            for i in xrange(list_len):
                self._placeholders[name][i] = self.__create_placeholder(
                    name + '_' + str(i) + '_',
                    dtype,
                    shape=shape,
                    sparse=sparse)
        else:
            self._placeholders[name] = self.__create_placeholder(
                name,
                dtype,
                shape=shape,
                sparse=sparse)

    def __create_placeholder(self, name, dtype, shape=None, sparse=False):
        if sparse:
            self.issparse.add(name)
            placeholder = dict()
            placeholder['indices'] = tf.placeholder(
                tf.int64, 
                [None, 2],
                name + '_indices')
            placeholder['values'] = tf.placeholder(
                dtype, 
                [None],
                name + '_values')
            placeholder['shape'] = tf.placeholder(
                tf.int64, 
                [2],
                name + '_shape')
            placeholder['tensor'] = tf.SparseTensor(
                placeholder['indices'],
                placeholder['values'],
                placeholder['shape'])
        else:
            placeholder = tf.placeholder(dtype, shape, name)

        return placeholder

    def __getitem__(self, name):
        """Returns placeholder with the given name.
        If name corresponds to a list of placeholders, the full list is
        returned. Similarly if name corresponds to a sparse placeholder a dict
        with keys 'indices', 'values', 'shape', and 'tensor' is returned. If
        a name corresponds to a list of sparse placeholders then a list of dicts
        (one for each sparse placeholder) is returned.
        Usage:
            plh_mgr = PlaceholderManager()
            plh_mgr.add_placeholder('var_name', tf.int64, sparse=True)
            placeholder = plh_mgr['var_name']
        """
        return self._placeholders[name]

    def get_feed_dict(self, inputs):
        """Returns a feed dictionary that can be passed to eval() or run().
        This method creates a feed dictionary from provided inputs that can be
        passed directly into eval() or run() routines in Tensorflow.
        Usage:
            import scipy.sparse as sps
            pm = PlaceholderManager()
            pm.add_placeholder('x', tf.float64, [1,2])
            pm.add_placeholder('y', tf.float64, [1,2])
            pm.add_placeholder('w', tf.float64, [1,2], list_len=2)
            pm.add_placeholder('I', tf.float64, [1,2], sparse=True)
            z = pm['x'] + pm['y'] + pm['w'][0] + pm['w'][1]
            z = tf.sparse_add(z,pm['I'])
        
            inputs = {
                'x': np.array([3.0, 4.0]),
                'y': np.array([5.0, 2.0]),
                'w': [np.array([1.0, 2.0]), np.array([3.0, 5.0])],
                'I': sps.eye(1,2)
            }
            feed_dict = pm.get_feed_dict(inputs)
            z.eval(feed_dict)
        Args:
            inputs (dict): A dictionary with placeholder names as keys and the
                inputs to be passed in as the values. For sparse placeholders
                only the sparse scipy matrix needs to be passed in instead of
                3 separate dense matrices of indices, values and shape. Inputs
                to a list of placeholders (list_len > 0) can be passed in as 
                a list.
        """
        feed_dict = dict()
        for name, input_value in inputs.items():
            if name in self.islist:
                for i in xrange(len(self._placeholders[name])):
                    self.__get_feed_dict_inner(
                        feed_dict, 
                        name + '_' + str(i) + '_',
                        self._placeholders[name][i], 
                        input_value[i])
            else:
                self.__get_feed_dict_inner(
                    feed_dict, 
                    name,
                    self._placeholders[name], 
                    input_value)

        return feed_dict

    def __get_feed_dict_inner(self, feed_dict, name, plh, input_value):
        if name in self.issparse:
            if isinstance(input_value, dict):
                input_value_ = input_value
            else:
                I, J, V = sps.find(input_value)
                input_value_ = dict()
                input_value_['indices'] = np.column_stack([I, J]).astype(
                    np.int64)
                input_value_['shape'] = np.array(input_value.shape).astype(
                    np.int64)
                if plh['values'].dtype == tf.int64:
                    input_value_['values'] = V.astype(np.int64)
                else:
                    input_value_['values'] = V
            for key in [k for k in plh.keys() if not k == 'tensor']:
                feed_dict[plh[key]] = input_value_[key]
        else:
            feed_dict[plh] = input_value

    def feed_dict_debug_string(self, feed_dict):
        """Returns feed dictionary as a neat string.
        Args:
            feed_dict (dict): Output of get_feed_dict() or a dictionary with
                placeholders as keys and the values to be fed into the graph as
                values.
        """
        debug_str = 'feed_dict={\n'
        feed_dict_plhs = [(plh, plh.name) for plh in feed_dict.keys()]
        feed_dict_plhs = sorted(feed_dict_plhs, key=lambda x: x[1])
        for plh, name in feed_dict_plhs:
            debug_str += '{}: \n{}\n'.format(plh, feed_dict[plh])
        debug_str += '}'
        return debug_str

    def placeholder_debug_string(self):
        """Returns placeholder information as a neat string.
        Args:
            placeholders (dict): Output of get_placeholders() or None in which
                case self._placeholders is used.
        """
        debug_str = pp.pformat(self._placeholders, indent=4, width=80)
        return debug_str

    def __str__(self):
        """Allows PlaceholderManager object to be used with print or str()."""
        return self.placeholder_debug_string()
