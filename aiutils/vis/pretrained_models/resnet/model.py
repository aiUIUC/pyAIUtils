# A convenient wrapper around ResNet based on 
# https://github.com/ry/tensorflow-resnet 
# See aiutils/examples/resnet_demo.py for example usage.
# The class supports -
# - Pretrained model initialization 
# - Average Pooling Layer (2048 dim) feature extraction
# - 1000 way object classification 

import inference as inference
from synset import *
from aiutils.tftools import var_collect
import numpy as np

import tensorflow as tf


def class_prediction(logits):
    """Top-5 class prediction from logits

    Args:
     - logits: 1D numpy array of logits

    Returns:
     - top5: list of top-5 class predictions
    """
    pred = np.argsort(logits)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print "Top1: ", top1

    # Get top5 label
    top5 = [synset[pred[i]] for i in range(5)]
    print "Top5: ", top5

    return top5


class ResnetInference():
    def __init__(
            self,
            images,
            color_mode='RGB',
            num_blocks=[3,4,6,3],
            class_layer=True,
            is_training=False,
            use_bias=False,
            bottleneck=True):
        """ResNet graph creation class.

        Args:
          - images: batch x h x w x 3 tensor containing the images with pixels
              in the range of [0., 255.]
          - color_mode (str): Either 'RGB' or 'BGR' specifying the input image 
              format provided
          - num_blocks (list): List of ints of length 4. num_blocks[i] is the
              number of resnet blocks at scale i+1 for i in range(1,5). Defaults
              to 50 layer resnet architecture
          - class_layer (bool): Attaches a 1000 way classification layer if True
          - is_training (bool): Training mode used for batch norm
          - use_bias (bool): Uses batch norm if set to False
          - bottleneck (bool): Uses bottleneck layers if set to True
        """
        self.graph = tf.get_default_graph()

        with tf.variable_scope('preprocess'):
            if color_mode=='RGB':
                images_ = tf.reverse(images, [False, False, False, True]) - \
                          inference.IMAGENET_MEAN_BGR
            elif color_mode=='BGR':
                images_ = images - IMAGENET_MEAN_BGR
            else:
                print 'Resnet expects input to be batch x h x w x 3 tensor ' + \
                    'with either BGR or RGB color channels ' + \
                    'and pixels in the range [0,255]'
                raise

        output = inference.inference(
            images_,
            is_training,
            num_classes=1000 if class_layer else None,
            num_blocks=num_blocks,
            use_bias=use_bias,
            bottleneck=bottleneck)

        if class_layer:
            self.logits = output
        else:
            self.logits=None

        self.avg_pool = self.graph.get_operation_by_name('avg_pool').outputs[0]

        self.resnet_vars = self.get_resnet_vars()
        self.restorer = self.create_restorer()

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

    def get_avg_pool_feat(self):
        """Returns average pooling layer features of dim batch_size x 2048.
        """
        return self.avg_pool

