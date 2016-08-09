# Demonstrates use of pretrained ResNet model available here:
# https://github.com/ry/tensorflow-resnet 

from aiutils.tftools import placeholder_management
from aiutils.vis import image, image_io
import aiutils.vis.pretrained_models.resnet.model as model
import os
import numpy as np

import tensorflow as tf
import pdb

if __name__=='__main__':
    im_h, im_w = (224, 224)
    
    # Path to the image to apply resnet on
    image_path = './aiutils/examples/telephone.jpg'
    
    # Checkpoint file to restore parameters from
    model_dir = '/home/tanmay/Downloads/pretrained_networks/' + \
                'Resnet/tensorflow-resnet-pretrained-20160509'
    ckpt_filename = os.path.join(model_dir, 'ResNet-L50.ckpt')

    # Graph construction
    graph = tf.Graph()
    with graph.as_default():
        plh = placeholder_management.PlaceholderManager()
        plh.add_placeholder(
            'images',
            tf.float32,
            shape=[None,im_h,im_w,3])

        resnet_model = model.ResnetInference(plh['images'])
    
    # Create feed dict
    im = image_io.imread(image_path)
    im = image.imresize(im, output_size=(im_h, im_w)).astype(np.float32)
    inputs = {
        'images': im.reshape(1, im_h, im_w, 3)
    }
    feed_dict = plh.get_feed_dict(inputs)
   
    # Restore model and get top-5 class predictions
    sess = tf.Session(graph=graph)
    resnet_model.restore_pretrained_model(sess, ckpt_filename)
    logits = resnet_model.logits.eval(feed_dict,sess)
    model.class_prediction(logits[0,:])
    pdb.set_trace()
    sess.close()
