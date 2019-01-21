from __future__ import division
import numpy as np
import tensorflow as tf

from skimage import io, color

def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    
    filter_size = get_kernel_size(factor)
    
    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)
    
    upsample_kernel = upsample_filt(filter_size)
    
    for i in range(number_of_classes):
        
        weights[:, :, i, i] = upsample_kernel
    
    return weights

# Input Tensor Shape: [batch_size, 28, 28, 128]
# Output Tensor Shape: [batch_size, 56, 56, 128]
# color_upsample1 = upsample.upsample_cnn(inputs=color_conv1, factor=2, number_of_classes=128, name="color_upsample1")
def upsample_cnn(inputs, factor, number_of_classes, size, name):
  with tf.name_scope(name):
      # factor = 2
      # number_of_classes = 2 #labels.shape[3]
      # new_height = labels.shape[1] * factor
      # new_width = labels.shape[2] * factor
      # expanded_img = labels
      # shape = tf.shape(inputs)
      # print("upsample_cnn() size: " + str(size))
      upsample_filter_np = bilinear_upsample_weights(factor, number_of_classes)
      # img_upsampled = tf.nn.conv2d_transpose(value=inputs, filter=upsample_filter_np,
      #                       output_shape=[shape[0], tf.multiply(shape[1],factor), tf.multiply(shape[2],factor), number_of_classes],
      #                       strides=[1, factor, factor, 1])
      img_upsampled = tf.nn.conv2d_transpose(value=inputs, filter=upsample_filter_np,
                            output_shape=[size[0], size[1], size[2], number_of_classes],
                            strides=[1, factor, factor, 1])
      return img_upsampled

def upsample_tf(factor, input_img):
    
    number_of_classes = input_img.shape[2]
    # print("input_img.shape: " + str(input_img.shape))
    # print("number_of_classes: " + str(number_of_classes))
    
    new_height = input_img.shape[0] * factor
    new_width = input_img.shape[1] * factor
    
    expanded_img = np.expand_dims(input_img, axis=0)
    # print("expanded_img: " + str(type(expanded_img)))
    # print("expanded_img: " + str(expanded_img.shape))

    with tf.Graph().as_default():
        with tf.Session() as sess:
            with tf.device("/cpu:0"):

                logits_pl = tf.placeholder(tf.float32)
                upsample_filt_pl = tf.placeholder(tf.float32)

                upsample_filter_np = bilinear_upsample_weights(factor,
                                        number_of_classes)
                # print("upsample_filter_np: " + str(upsample_filter_np.shape))

                res = tf.nn.conv2d_transpose(value=logits_pl, filter=upsample_filt_pl,
                        output_shape=[1, new_height, new_width, number_of_classes],
                        strides=[1, factor, factor, 1])

                final_result = sess.run(res,
                                feed_dict={upsample_filt_pl: upsample_filter_np,
                                           logits_pl: expanded_img})
    
    return final_result.squeeze()

# filename = "./dataset/images/000b5757932a8050337257fdae2c9942.jpg"
# image_rgb = io.imread(filename)
# upsampled_img_tf = upsample_tf(factor=3, input_img=image_rgb)
# print("upsampled_img_tf: " + str(type(upsampled_img_tf)))
# print("upsampled_img_tf: " + str(upsampled_img_tf.shape))
# # print("upsampled_img_tf: " + str(upsampled_img_tf[0,0,:]))
# upsampled_img_tf = upsampled_img_tf.astype(np.uint8)
# # io.imshow(upsampled_img_tf)
# io.imsave("./output/tmp.jpg", upsampled_img_tf)