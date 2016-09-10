# Base class for resnet based models

import inference as inference
from synset import *
from aiutils.tftools import var_collect
import numpy as np

import tensorflow as tf


class ResnetInferenceBase(object):
    def __init__(
            self,
            images,
            color_mode='RGB'):
        """ResNet graph creation base class.

        Args:
          - images: batch x h x w x 3 tensor containing the images with pixels
              in the range of [0., 255.]
          - color_mode (str): Either 'RGB' or 'BGR' specifying the input image 
              format provided
        """
        self.graph = tf.get_default_graph()

        with tf.variable_scope('preprocess'):
            if color_mode=='RGB':
                self.images_ = tf.reverse(images, [False, False, False, True]) - \
                          inference.IMAGENET_MEAN_BGR
            elif color_mode=='BGR':
                self.images_ = images - IMAGENET_MEAN_BGR
            else:
                print 'Resnet expects input to be batch x h x w x 3 tensor ' + \
                    'with either BGR or RGB color channels ' + \
                    'and pixels in the range [0,255]'
                raise

    def get_resnet_vars(self):
        """Returns list of resnet variables.
        """
        resnet_vars = []
        for i in xrange(5):
            resnet_vars += var_collect.collect_scope('scale' + str(i+1))

        resnet_vars += var_collect.collect_scope('fc')

        return resnet_vars
        
    def create_restorer(self):
        """Creates a Saver that can be used to save or restore resnet vars.
        """
        return tf.train.Saver(self.resnet_vars)

    def restore_pretrained_model(self, sess, ckpt_path):
        """Restores variables from specified checkpoint file.
        
        Args:
          - sess (tf.Session): session to restore the variables in 
          - ckpt_path (string): path to resnet checkpoint
        """
        self.restorer.restore(sess, ckpt_path)

    def get_logits(self):
        """Returns logits of dimension batch_size x 1000.
        """
        return self.logits

    def get_batchnorm_update_ops(self):
        """Returns batchnorm update ops that need to be grouped with train_op
        """
        resnet_bn_updates = tf.get_default_graph().get_collection(
            'resnet_update_ops')
        return tf.group(*resnet_bn_updates)
 
    def imagenet_class_prediction(self, logits, k=5, verbose=True):
        """Top-k class prediction from logits

        Args:
         - logits: 1D numpy array of logits
         - k: number of top predictions to return
         - verbose: prints the predictions when set to True

        Returns:
         - top_k: list of top-k class predictions
        """
        pred = np.argsort(logits)[::-1]

        # Get top k label
        top_k = [synset[pred[i]] for i in range(k)]

        if verbose: 
            print "Top {}: {}".format(k, top_k)

        return top_k
        
