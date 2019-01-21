
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import upsample_tf_1_3 as upsample
import pix2pix

from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean
import scipy

tf.logging.set_verbosity(tf.logging.INFO)

BATCH_SIZE = 16
STEPS = 100000
LR = 1.

def fusion_map(_global, mid, w, b):
  # print("global: " + str(_global))
  # print("mid: " + str(mid))
  # out [784,512]
  concat = tf.concat([_global, mid], 1)
  # print("concat: " + str(concat))
  # res = tf.map_fn(lambda y_mid: fusion(y_global, y_mid, w, b), mid)
  res = tf.add(tf.matmul(concat, w), b)
  # y_mid = tf.slice(mid, [0, 0], [1, 256]) # shape [1, 256]
  # print("y_mid: " + str(y_mid))

  # y_global = tf.slice(_global, [0, 0], [1, 256])
  # res = tf.Variable([], dtype=tf.float32)
  # for i in range(28*28):
  #   #lab_l = tf.slice(lab1, [0, 0, 0], [224, 224, 1])
  #   y_mid = tf.slice(mid, [i, 0], [1, 256])
  #   # y_mid = tf.reshape(y_mid, [256])
  #   seg = fusion(y_global, y_mid, w, b)
  #   res = tf.concat([res, seg], 0)

  # res = tf.reshape(res, [28*28, 256])
  # print("fusion_map() res: " + str(res))

  return res

def duplicate_global(segment):
  list = []
  # list += [segment]
  # zeros = tf.zeros([256], dtype=tf.float32)
  for i in range(28*28):
    list += [segment]
  return tf.stack(list)
  # result = tf.Variable([], dtype=tf.float32)
  # for i in range(28*28):
  #   result = tf.concat([result, segment], 0)

  # return tf.reshape(result, [28*28, 256])

# input_global [-1,256], input_mid [-1,28,28,256], output [-1,28,28,256]
def fusion_layer(input_global, input_mid):
  with tf.name_scope("fusion"):
    w = tf.Variable(tf.truncated_normal([512, 256], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[28*28, 256]), name="B")

    # out [-1,28,28,256]
    # print("input_global: " + str(input_global))
    global_dup = tf.map_fn(lambda segment: duplicate_global(segment), input_global)
    # print("global_dup: " + str(global_dup))

    # # # out [-1,28,28,512]
    # # concat = tf.concat([global_dup, input_mid], 3)

    # # conv = tf.nn.conv2d(concat, w, strides=[1, 1, 1, 1], padding="SAME")
    # # act = tf.nn.sigmoid(conv + b)

    input_mid = tf.reshape(input_mid, [-1, 28*28, 256])

    combine = (global_dup, input_mid)

    res = tf.map_fn(lambda x: fusion_map(x[0], x[1], w, b), combine, dtype=tf.float32)
    # res = tf.Variable([], dtype=tf.float32)
    # for i in range(BATCH_SIZE):
    #   one_global = tf.slice(input_global, [i, 0], [1, 256])
    #   one_mid = tf.slice(input_mid, [i, 0, 0], [1, 28*28, 256])
    # i = tf.constant(0)
    # c = lambda i: tf.less(i, BATCH_SIZE)
    # b = lambda i: tf.add(i, 1)
    # r = tf.while_loop(c, b, [i, input_global, input_mid])

    # return tf.reshape(input_mid, [-1, 28, 28, 256])
    return tf.reshape(res, [-1, 28, 28, 256])

def color_cnn_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # Input images are 224x224 pixels, and have one color channel
  # print("features[x]: " + str(features["x"]))
  # print("labels: " + str(labels))
  # input_layer = tf.reshape(features["x"], [-1, 224, 224, 1])
  input_layer = features["x"]
  # input_layer = tf.cast(input_layer, dtype=tf.float32)
  # tf.image.per_image_standardization(lab_l) tf.nn.l2_normalize
  tf.summary.image('colornet_input', input_layer, 3)
  tf.summary.image('colornet_truth', features["truth"], 3)
  # one_image = tf.slice(features["truth"], [0, 0, 0, 0], [1, 224, 224, 3])
  # one_image = tf.reshape(one_image, [224, 224, 3])
  # tf.summary.image('per_image_standardization', tf.reshape(tf.image.per_image_standardization(one_image), [1, 224, 224, 3]), 1)
  # tf.summary.image('batch_normalization', tf.layers.batch_normalization(features["truth"]), 1)
  # tf.summary.image('l2_normalize', tf.nn.l2_normalize(features["truth"], None), 1)

  # input_normalized = tf.image.per_image_standardization(input_layer)
  input_normalized = tf.map_fn(lambda image: tf.image.per_image_standardization(image), input_layer)
  # input_normalized = tf.layers.batch_normalization(input_layer)

  # Low-Level Features Network
  # Convolutional Layer #1
  # Computes 64 features using a 3x3 filter with Sigmoid activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 224, 224, 1]
  # Output Tensor Shape: [batch_size, 112, 112, 64]
  ll_conv1 = tf.layers.conv2d(
      inputs=input_normalized,
      filters=64,
      kernel_size=[3, 3],
      strides=(2, 2),
      padding="same",
      activation=tf.nn.relu,
      name="ll_conv1")
  # ll_conv1 = tf.layers.batch_normalization(ll_conv1)

  # Input Tensor Shape: [batch_size, 112, 112, 64]
  # Output Tensor Shape: [batch_size, 112, 112, 128]
  ll_conv2 = tf.layers.conv2d(
      inputs=ll_conv1,
      filters=128,
      kernel_size=[3, 3],
      strides=(1, 1),
      padding="same",
      activation=tf.nn.relu,
      name="ll_conv2")
  # ll_conv2 = tf.layers.batch_normalization(ll_conv2)

  # Input Tensor Shape: [batch_size, 112, 112, 128]
  # Output Tensor Shape: [batch_size, 56, 56, 128]
  ll_conv3 = tf.layers.conv2d(
      inputs=ll_conv2,
      filters=128,
      kernel_size=[3, 3],
      strides=(2, 2),
      padding="same",
      activation=tf.nn.relu,
      name="ll_conv3")
  # ll_conv3 = tf.layers.batch_normalization(ll_conv3)

  # Input Tensor Shape: [batch_size, 56, 56, 128]
  # Output Tensor Shape: [batch_size, 56, 56, 256]
  ll_conv4 = tf.layers.conv2d(
      inputs=ll_conv3,
      filters=256,
      kernel_size=[3, 3],
      strides=(1, 1),
      padding="same",
      activation=tf.nn.relu,
      name="ll_conv4")
  # ll_conv4 = tf.layers.batch_normalization(ll_conv4)

  # Input Tensor Shape: [batch_size, 56, 56, 256]
  # Output Tensor Shape: [batch_size, 28, 28, 256]
  ll_conv5 = tf.layers.conv2d(
      inputs=ll_conv4,
      filters=256,
      kernel_size=[3, 3],
      strides=(2, 2),
      padding="same",
      activation=tf.nn.relu,
      name="ll_conv5")
  # ll_conv5 = tf.layers.batch_normalization(ll_conv5)

  # Input Tensor Shape: [batch_size, 28, 28, 256]
  # Output Tensor Shape: [batch_size, 28, 28, 512]
  ll_conv6 = tf.layers.conv2d(
      inputs=ll_conv5,
      filters=512,
      kernel_size=[3, 3],
      strides=(1, 1),
      padding="same",
      activation=tf.nn.relu,
      name="ll_conv6")
  # ll_conv6 = tf.layers.batch_normalization(ll_conv6)

  # Mid-Level Features Network
  # Input Tensor Shape: [batch_size, 28, 28, 512]
  # Output Tensor Shape: [batch_size, 28, 28, 512]
  ml_conv1 = tf.layers.conv2d(
      inputs=ll_conv6,
      filters=512,
      kernel_size=[3, 3],
      strides=(1, 1),
      padding="same",
      activation=tf.nn.relu,
      name="ml_conv1")
  # ml_conv1 = tf.layers.batch_normalization(ml_conv1)

  # Input Tensor Shape: [batch_size, 28, 28, 512]
  # Output Tensor Shape: [batch_size, 28, 28, 256]
  ml_conv2 = tf.layers.conv2d(
      inputs=ml_conv1,
      filters=256,
      kernel_size=[3, 3],
      strides=(1, 1),
      padding="same",
      activation=tf.nn.relu,
      name="ml_conv2")
  # ml_conv2 = tf.layers.batch_normalization(ml_conv2)

  # Global Features Network
  # Input Tensor Shape: [batch_size, 28, 28, 512]
  # Output Tensor Shape: [batch_size, 28, 28, 256]
  global_conv = tf.layers.conv2d(
      inputs=ll_conv6,
      filters=2,
      kernel_size=[3, 3],
      strides=(1, 1),
      padding="same",
      activation=tf.nn.relu,
      name="global_conv")

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 28, 28, 512]
  # Output Tensor Shape: [batch_size, 28 * 28 * 512]
  global_flat = tf.reshape(global_conv, [-1, 28 * 28 * 2], name="global_flat")

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 28 * 28 * 512]
  # Output Tensor Shape: [batch_size, 1024]
  global_dense1 = tf.layers.dense(inputs=global_flat, units=1024, activation=tf.nn.relu, name="global_dense1")

  global_dense2 = tf.layers.dense(inputs=global_dense1, units=512, activation=tf.nn.relu, name="global_dense2")

  global_dense3 = tf.layers.dense(inputs=global_dense2, units=256, activation=tf.nn.relu, name="global_dense3")

  # global_dense3: [batch_size, 256], ml_conv2: [batch_size, 28, 28, 256]
  fusion = fusion_layer(input_global=global_dense3, input_mid=ml_conv2)

  # Colorization Network
  # Input Tensor Shape: [batch_size, 28, 28, 256]
  # Output Tensor Shape: [batch_size, 28, 28, 128] ml_conv2
  color_conv1 = tf.layers.conv2d(
      inputs=fusion,
      filters=128,
      kernel_size=[3, 3],
      strides=(1, 1),
      padding="same",
      activation=tf.nn.relu,
      name="color_conv1")
  # color_conv1 = tf.layers.batch_normalization(color_conv1)
  # print("color_conv1: " + str(color_conv1))

  # Input Tensor Shape: [batch_size, 28, 28, 128]
  # Output Tensor Shape: [batch_size, 56, 56, 128]
  color_upsample1 = upsample.upsample_cnn(inputs=color_conv1, factor=2, number_of_classes=128, size=(BATCH_SIZE, 56, 56), name="color_upsample1")
  # color_upsample1 = tf.layers.batch_normalization(color_upsample1)
  # tf.assert_equal(tf.shape(color_upsample1)[0], BATCH_SIZE)

  # Input Tensor Shape: [batch_size, 56, 56, 128]
  # Output Tensor Shape: [batch_size, 56, 56, 64]
  color_conv2 = tf.layers.conv2d(
      inputs=color_upsample1,
      filters=64,
      kernel_size=[3, 3],
      strides=(1, 1),
      padding="same",
      activation=tf.nn.relu,
      name="color_conv2")
  # color_conv2 = tf.layers.batch_normalization(color_conv2)

  # Input Tensor Shape: [batch_size, 56, 56, 64]
  # Output Tensor Shape: [batch_size, 56, 56, 64]
  color_conv3 = tf.layers.conv2d(
      inputs=color_conv2,
      filters=64,
      kernel_size=[3, 3],
      strides=(1, 1),
      padding="same",
      activation=tf.nn.relu,
      name="color_conv3")
  # color_conv3 = tf.layers.batch_normalization(color_conv3)

  # Input Tensor Shape: [batch_size, 56, 56, 64]
  # Output Tensor Shape: [batch_size, 112, 112, 64]
  color_upsample2 = upsample.upsample_cnn(inputs=color_conv3, factor=2, number_of_classes=64, size=(BATCH_SIZE, 112, 112), name="color_upsample2")
  # color_upsample2 = tf.layers.batch_normalization(color_upsample2)
  # tf.assert_equal(tf.shape(color_upsample2)[0], BATCH_SIZE)

  # Input Tensor Shape: [batch_size, 112, 112, 64]
  # Output Tensor Shape: [batch_size, 112, 112, 32]
  color_conv4 = tf.layers.conv2d(
      inputs=color_upsample2,
      filters=32,
      kernel_size=[3, 3],
      strides=(1, 1),
      padding="same",
      activation=tf.nn.relu,
      name="color_conv4")
  # color_conv4 = tf.layers.batch_normalization(color_conv4)

  # Input Tensor Shape: [batch_size, 112, 112, 32]
  # Output Tensor Shape: [batch_size, 112, 112, 2]
  output_conv = tf.layers.conv2d(
      inputs=color_conv4,
      filters=2,
      kernel_size=[3, 3],
      strides=(1, 1),
      padding="same",
      activation=tf.nn.sigmoid,
      name="output_conv")
  # output_conv = tf.layers.batch_normalization(output_conv)
  # if tf.shape(output_conv)[0] != BATCH_SIZE:
    # labels = tf.slice(labels, [0, 0, 0, 0], [tf.shape(output_conv)[0], 112, 112, 2])
    # labels = tf.zeros(tf.shape(labels), dtype=tf.float32)
    # x = tf.Variable()
    # output_conv = tf.ones(tf.shape(labels), dtype=tf.float32)
    # output_conv = tf.scalar_mul(0.5, output_conv)
    # output_conv = tf.Variable(tf.truncated_normal([BATCH_SIZE, 112, 112, 2], stddev=0.1), dtype=tf.float32)

  # Input Tensor Shape: [batch_size, 112, 112, 2]
  # Output Tensor Shape: [batch_size, 224, 224, 2]
  # input_resized = tf.image.resize_images(input_layer, [112, 112])
  output_upsample = upsample.upsample_cnn(inputs=output_conv, factor=2, number_of_classes=2, size=(BATCH_SIZE, 224, 224), name="output_upsample")

  # Output Tensor Shape: [batch_size, 224, 224, 3]
  # output_images = input_layer + output_conv
  with tf.name_scope("get_output_image"):
    output_denormalized = lab_logit(output_upsample)
    output_lab = tf.concat([input_layer, output_denormalized], 3)
    output_rgb = pix2pix.lab_to_rgb(output_lab)

    tf.summary.image('colornet_output', output_rgb, 3)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "predict": tf.identity(output_rgb, name="output_rgb"),
      "truth": features["truth"]
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  def func1():
    return tf.losses.mean_squared_error(labels=labels, predictions=output_conv)
  def func2():
    return tf.constant(0.03111111, dtype=tf.float32)
  loss = tf.cond(tf.equal(tf.shape(output_conv)[0], BATCH_SIZE), true_fn=func1, false_fn=func2)
  # loss = tf.losses.mean_squared_error(labels=labels, predictions=output_conv)
  # loss = tf.losses.mean_pairwise_squared_error(labels=labels, predictions=output_conv)
  # loss_100 = tf.multiply(loss, 100., name="loss_100")
  # tf.summary.histogram("loss_100", loss_100)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    # global_step = tf.Variable(0, trainable=False)
    global_step = tf.train.get_global_step()
    base_learning_rate = 0.00001 * (LR) # 0.00001
    learning_rate = tf.train.exponential_decay(base_learning_rate, global_step,
                                               160000, 0.316, staircase=True)

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": None}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# linear normalization for rgb channels of images, [0, 255] to [0, 1]
def rgb_normalize(image):
  tmp_image = tf.cast(image, tf.float32)
  
  return tf.scalar_mul(1.0 / 255, tmp_image)

# based on dataset, offset according to percentage of positive over nagative value
deltaX = 0.565714 - 0.5
# non-linear normalization for a*b* channels of lab images
# L [0, 100] ab [-110, 100], just for ab, f(x) = 1 / (1 + e^(-(x/10-deltaX)))
def lab_sigmoid(image):

  ones = tf.ones(image.shape, dtype=tf.float32)
  
  image = tf.scalar_mul(1.0 / 10, image)
  image = tf.subtract(image, tf.scalar_mul(deltaX, ones))
  
  return tf.sigmoid(image)

# non-linear denormalization for a*b* channels of lab images
# f(y) = 10 * (ln(y / (1 - y)) + deltaX)
def lab_logit(image):
  ones = tf.ones(tf.shape(image), dtype=tf.float32)
  
  image = tf.py_func(scipy.special.logit, [image], tf.float32)
  image = tf.add(image, tf.scalar_mul(deltaX, ones))

  return tf.scalar_mul(10.0, image)

def _parse_fn(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3) #unit8, 0-255

  image_norm = rgb_normalize(image_decoded) #float32, 0-1
  image_resized1 = tf.image.resize_images(image_norm, [224, 224]) #float32, 0-1
  image_resized2 = tf.image.resize_images(image_norm, [112, 112])
  
  lab1 = pix2pix.rgb_to_lab(image_resized1)
  lab2 = pix2pix.rgb_to_lab(image_resized2)

  lab_l = tf.slice(lab1, [0, 0, 0], [224, 224, 1])
  lab_ab = tf.slice(lab2, [0, 0, 1], [112, 112, 2])
  
  # label gray images, which max(lab_ab) < 1 and min(lab_ab) > -1, as not valid image
  max = tf.reduce_max(lab_ab)
  min = tf.reduce_min(lab_ab)
  def func1(): 
    return {"x": lab_l, "truth": image_resized1, "valid": tf.constant(False)}, tf.zeros(tf.shape(lab_ab), dtype=tf.float32)
  def func2(): 
    return {"x": lab_l, "truth": image_resized1, "valid": tf.constant(True)}, lab_sigmoid(lab_ab)
    
  return tf.cond(tf.logical_and(tf.less(max, 1.), tf.greater(min, -1.)), true_fn=func1, false_fn=func2)

def train_input_fn():
  image_path = "./dataset/images/"
  image_names = []
  # get all the image names in train.txt for training
  file = open("./dataset/train.txt", "r")
  for line in file:
      image_names += [image_path + line.replace("\n", "")]
  file.close()
  file = open("./dataset/test.txt", "r")
  for line in file:
      image_names += [image_path + line.replace("\n", "")]
  file.close()

  # convert list to tensor
  filenames = tf.constant(image_names)

  # create a temporary labels for later usage
  labels = tf.constant(np.zeros(len(image_names)))
  labels = tf.cast(labels, tf.float32)

  # create a dataset to manage the input images and labels
  dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))

  # Use `Dataset.map()` to build a pair of a feature dictionary and a label
  dataset = dataset.map(_parse_fn)
  # filter out invalid images (gray images)
  dataset = dataset.filter(lambda x, _ : x["valid"])
  # set buffer size for suffling
  dataset = dataset.shuffle(buffer_size=500)
  # set batch size
  dataset = dataset.batch(BATCH_SIZE)
  # set number of epochs for dataset
  dataset = dataset.repeat(999999)
  # print("dataset: " + str(dataset))
  # build a one_shot_iterator for iteration
  iterator = dataset.make_one_shot_iterator()

  # `features` is a dictionary in which each value is a batch of values for
  # that feature; `labels` is a batch of labels.
  features, labels = iterator.get_next()
  return features, labels

def eval_input_fn():
  image_path = "./dataset/images/"
  image_names = []
  # get all the image names in train.txt for training
  file = open("./dataset/test.txt", "r")
  i = 0
  for line in file:
    image_names += [image_path + line.replace("\n", "")]
    if i > 100:
      break
    i += 1

  file.close()

  # convert list to tensor
  filenames = tf.constant(image_names)

  # create a temporary labels for later usage
  labels = tf.constant(np.zeros(len(image_names)))
  labels = tf.cast(labels, tf.float32)

  # create a dataset to manage the input images and labels
  dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))

  # Use `Dataset.map()` to build a pair of a feature dictionary and a label
  dataset = dataset.map(_parse_fn)
  # set buffer size for suffling
  # dataset = dataset.shuffle(buffer_size=100)
  # set batch size
  dataset = dataset.batch(BATCH_SIZE)
  # set number of epochs for dataset
  # dataset = dataset.repeat(999999)
  # print("dataset: " + str(dataset))
  # build a one_shot_iterator for iteration
  iterator = dataset.make_one_shot_iterator()

  # `features` is a dictionary in which each value is a batch of values for
  # that feature; `labels` is a batch of labels.
  features, labels = iterator.get_next()
  return features, labels

def main(unused_argv):

  print("Colorization main()")

  # Create the Estimator
  #cnn_model_fn color_cnn_fn
  color_painter = tf.estimator.Estimator(
      model_fn=color_cnn_fn, model_dir="./tmp/color_global_batch" + str(BATCH_SIZE))

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  # tensors_to_log = {"probabilities": "softmax_tensor"}
  # tensors_to_log = {"images": "output_rgb"}
  # tensors_to_log = {"loss_100": "loss_100"}
  # logging_hook = tf.train.LoggingTensorHook(
  #     tensors=tensors_to_log, every_n_iter=100)

  # Train the model
  color_painter.train(
      input_fn=train_input_fn,
      steps=STEPS,
      hooks=None)

  # # Evaluate the model and print results
  # eval_results = color_painter.evaluate(input_fn=eval_input_fn)
  # print(eval_results)

  # # Predict by using the model and print results
  # predict_image_path = "./predict/"
  # predict_results = color_painter.predict(input_fn=eval_input_fn)
  # print(type(predict_results))
  # print(type(predict_results.next()))
  # print(type(predict_results.next()["predict"]))
  # print(predict_results.next()["predict"].shape)
  # for i in range(1, 11):
  #   next_predict = predict_results.next()
  #   io.imsave(predict_image_path + "predict"+ str(i) +".jpg", next_predict["predict"])
  #   io.imsave(predict_image_path + "truth"+ str(i) +".jpg", next_predict["truth"])


if __name__ == "__main__":
  tf.app.run()