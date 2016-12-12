import tensorflow as tf

def resize_images_like(images, reference_tensor, method=0, align_corners=False):
    """ Image resize helper.

    Resizes images to the same size as images_like.

    Args:
      images (4d tensor): Input to resize layer
        Should have shape `[batch, width, height, channels]`

      reference_tensor (4d tensor): defines shape to resize to
        Should have shape `[?, width_new, height_new, ?]`
      method: ResizeMethod.  Defaults to `ResizeMethod.BILINEAR`.
      align_corners: bool. If true, exactly align all 4 cornets of the input and
         output. Defaults to `false`.

    Returns:
      output (tensor): The resized image.
        Will have shape `[batch, width_new, height_new, channels]`
    """
    _,w,h,_ = reference_tensor.get_shape().as_list()
    return tf.image.resize_images(images, [w, h], method=method, align_corners=align_corners)
