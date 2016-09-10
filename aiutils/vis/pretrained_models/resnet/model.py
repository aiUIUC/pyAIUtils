# A convenient wrapper around ResNet based on 
# https://github.com/ry/tensorflow-resnet 
# See aiutils/examples/resnet_demo.py for example usage.
# For a full list of supported methods see the base class in
# aiutils/vis/pretrained_models/resnet/model_base.py
import inference
import model_base

import tensorflow as tf


class ResnetInference(model_base.ResnetInferenceBase):
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
        super(ResnetInference, self).__init__(
            images,
            color_mode)

        output = inference.inference(
            self.images_,
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

    def get_avg_pool_feat(self):
        """Returns average pooling layer features of dim batch_size x 2048.
        """
        return self.avg_pool



