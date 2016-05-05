import tensorflow as tf
import numpy as np
import scipy.sparse as sps

 
class PlholderManager():
    def __init__(self):
        self.plholders = dict()
        self.issparse = dict()
    
    def add_plholder(self, name, dtype, shape=None, sparse=False):
        self.issparse[name] = sparse
        if not sparse:
            self.plholders[name] = tf.placeholder(dtype, shape, name)

        else:
            name_indices = name + '_indices'
            name_values = name + '_values'
            name_shape = name + '_shape'

            self.plholders[name_indices] = tf.placeholder(tf.int64, [None, 2], 
                                                          name_indices)
            self.plholders[name_values] = tf.placeholder(dtype, [None], 
                                                         name_values)
            self.plholders[name_shape] = tf.placeholder(tf.int64, [2], 
                                                        name_shape)

    def get_proxy_plholders(self):
        proxy_plholders = dict()
        for name, sparse in self.issparse.items():
            if not sparse:
                proxy_plholders[name] = self.plholders[name]
            else:
                plholder_indices = self.plholders[name + '_indices']
                plholder_values = self.plholders[name + '_values']
                plholder_shape = self.plholders[name + '_shape']
                sparse_tensor = tf.SparseTensor(plholder_indices, 
                                                plholder_values,
                                                plholder_shape)  
                proxy_plholders[name] = sparse_tensor

        return proxy_plholders

    def get_feed_dict(self, inputs):
        feed_dict = dict()
        for name, input_value in self.inputs.items():
            try:
                plholder_sparsity = self.issparse[name]
                input_sparsity = sps.issparse(input_value)
                assert_str = 'Sparsity of placeholder and input do not match'
                assert (plholder_sparsity == input_sparsity), assert_str
                if not input_sparsity:
                    plholder = self.plholders[name]
                    feed_dict[plholder] = input_value

                else:
                    I,J,V = find(input_value)

                    plholder_indices = self.plholders[name + '_indices']
                    plholder_values = self.plholders[name + '_values']
                    plholder_shape = self.plholders[name + '_shape']

                    feed_dict[plholder_indices] = \
                        np.column_stack(I, J).astype(np.int64)
                    feed_dict[plholder_values] = V
                    feed_dict[plholder_shape] = \
                        np.array(input_value.shape).astype(np.int64)

            except KeyError:
                print "No placeholder with name '{}'".format(name)
                raise
                
        return feed_dict


                
