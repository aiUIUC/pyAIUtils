import tensorflow as tf

def resize_images_like(images, images_like, method=0, align_corners=False):
    """ Image resize helper.

    Resizes images to the same size as images_like.

    Args:
      images (4d tensor): Input to resize layer
        Should have shape `[batch, width, height, channels]`

      images_like (4d tensor): defines shape to resize to
        Should have shape `[?, width_new, height_new, ?]`

      other arguments see tf.image.resize_images

    Returns:
      output (tensor): The resized image.
        Will have shape `[batch, width_new, height_new, channels]`
    """
    _,w,h,_ = images_like.get_shape().as_list()
    return tf.image.resize_images(images, w, h, method, align_corners)
