
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
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

def color_cnn_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # Input images are 224x224 pixels, and have one color channel
  
  input_layer = features["x"]

  tf.summary.image('colornet_input', input_layer, 3)
  tf.summary.image('colornet_truth', features["truth"], 3)

  # input_normalized = tf.image.per_image_standardization(input_layer)
  input_normalized = tf.map_fn(lambda image: tf.image.per_image_standardization(image), input_layer)
  # input_normalized = tf.layers.batch_normalization(input_layer)
  # print("input_normalized: " + str(input_normalized))

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

  # Colorization Network
  # Input Tensor Shape: [batch_size, 28, 28, 256]
  # Output Tensor Shape: [batch_size, 28, 28, 128] ml_conv2
  color_conv1 = tf.layers.conv2d(
      inputs=ml_conv2,
      filters=128,
      kernel_size=[3, 3],
      strides=(1, 1),
      padding="same",
      activation=tf.nn.relu,
      name="color_conv1")
  # color_conv1 = tf.layers.batch_normalization(color_conv1)

  # Input Tensor Shape: [batch_size, 28, 28, 128]
  # Output Tensor Shape: [batch_size, 56, 56, 128]
  color_upsample1 = upsample.upsample_cnn(inputs=color_conv1, factor=2, number_of_classes=128, size=(BATCH_SIZE, 56, 56), name="color_upsample1")
  # color_upsample1 = tf.layers.batch_normalization(color_upsample1)
  tf.assert_equal(tf.shape(color_upsample1)[0], BATCH_SIZE)

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
  tf.assert_equal(tf.shape(color_upsample2)[0], BATCH_SIZE)

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
  #   return tf.estimator.EstimatorSpec(mode=mode, train_op=None)

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
      "images": tf.identity(output_rgb, name="output_rgb")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.mean_squared_error(labels=labels, predictions=output_conv)
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

def rgb_normalize(image):
  tmp_image = tf.cast(image, tf.float32)
  
  return tf.scalar_mul(1.0 / 255, tmp_image)

# based on dataset, offset according to percentage of positive over nagative value
deltaX = 0.565714 - 0.5
# L [0, 100] ab [-110, 100], just for ab, 1 / (1 + e^(-(x/10-deltaX)))
def lab_sigmoid(image):

  ones = tf.ones(tf.shape(image), dtype=tf.float32)
  
  image = tf.scalar_mul(1.0 / 10, image)
  image = tf.subtract(image, tf.scalar_mul(deltaX, ones))
  
  return tf.sigmoid(image)

# 10 * (ln(y / (1 - y)) + deltaX)
def lab_logit(image):
  ones = tf.ones(tf.shape(image), dtype=tf.float32)
  
  image = tf.py_func(scipy.special.logit, [image], tf.float32)
  image = tf.add(image, tf.scalar_mul(deltaX, ones))

  return tf.scalar_mul(10.0, image)

def _parse_fn(filename, label):
  image_string = tf.read_file(filename)
  # image_decoded = tf.image.decode_image(image_string)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3) #unit8, 0-255
  # print("image_decoded: " + str(image_decoded))
  image_norm = rgb_normalize(image_decoded) #float32, 0-1
  image_resized1 = tf.image.resize_images(image_norm, [224, 224]) #float32, 0-1
  image_resized2 = tf.image.resize_images(image_norm, [112, 112])
  
  # tf.summary.image('image_resized1', tf.reshape(image_resized1, [1,224,224,3]), 3)
  # print("image_resized1: " + str(image_resized1))

  lab1 = pix2pix.rgb_to_lab(image_resized1)
  lab2 = pix2pix.rgb_to_lab(image_resized2)
  # print("lab1: " + str(lab1))

  lab_l = tf.slice(lab1, [0, 0, 0], [224, 224, 1])
  lab_ab = tf.slice(lab2, [0, 0, 1], [112, 112, 2])
  # lab_ab = tf.slice(lab1, [0, 0, 1], [224, 224, 2])
  # remove gray images, max < 1 and min > -1
  max = tf.reduce_max(lab_ab)
  min = tf.reduce_min(lab_ab)
  def func1(): 
    return {"x": lab_l, "truth": image_resized1, "valid": tf.constant(False)}, tf.zeros(tf.shape(lab_ab), dtype=tf.float32)
  def func2(): 
    return {"x": lab_l, "truth": image_resized1, "valid": tf.constant(True)}, lab_sigmoid(lab_ab)
    
  return tf.cond(tf.logical_and(tf.less(max, 1.), tf.greater(min, -1.)), true_fn=func1, false_fn=func2)

def dataset_input_fn():
  image_path = "./dataset/images/"
  image_names = []
  num_epochs = 999999
  file = open("./dataset/train.txt", "r")
  for line in file:
      image_names += [image_path + line.replace("\n", "")]
  file.close()
  file = open("./dataset/test.txt", "r")
  for line in file:
      image_names += [image_path + line.replace("\n", "")]
  file.close()

  # A vector of filenames.
  filenames = tf.constant(image_names)

  # `labels[i]` is the label for the image in `filenames[i].
  labels = tf.constant(np.zeros(len(image_names)))
  labels = tf.cast(labels, tf.float32)
  print("labels: " + str(labels))

  dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))

  # Use `Dataset.map()` to build a pair of a feature dictionary and a label
  # tensor for each example. _parse_fn map_all_zero
  dataset = dataset.map(_parse_fn)
  # dataset = dataset.filter( lambda v : v <= 6 )
  # remove gray images, invalid input, max < 1 and min > -1
  # dataset = dataset.filter(lambda _, x : tf.logical_or(tf.reduce_max(x) > 1, tf.reduce_min(x) < -1))
  # dataset = dataset.filter(lambda _, x : tf.reduce_sum(x) < 0.0001)
  dataset = dataset.filter(lambda x, _ : x["valid"])
  dataset = dataset.shuffle(buffer_size=500)
  dataset = dataset.batch(BATCH_SIZE)
  dataset = dataset.repeat(num_epochs)
  print("dataset: " + str(dataset))
  iterator = dataset.make_one_shot_iterator()

  # `features` is a dictionary in which each value is a batch of values for
  # that feature; `labels` is a batch of labels.
  features, labels = iterator.get_next()
  return features, labels

def usage():
  print("Usage:")
  print("    python color_tf_1_3.py train")
  print("    python color_tf_1_3.py eval")
  print("    python color_tf_1_3.py predict")
  print("    python color_tf_1_3.py train <model_dir> <batch_size>")
  print("    python color_tf_1_3.py eval <model_dir> <batch_size>")
  print("    python color_tf_1_3.py predict <model_dir> <predict_image_dir>")
  print("    model_dir: the directory where the model will be saved at")
  print("Default:")
  print("    python color_tf_1_3.py train ./tmp/color_batch<BATCH_SIZE> 16")
  print("    python color_tf_1_3.py eval ./tmp/color_batch<BATCH_SIZE> 16")
  print("    python color_tf_1_3.py predict ./tmp/color_batch<BATCH_SIZE> ./predict/")

def main(argv):

  print("Basic Colorization main()")
  # check command line arguments 
  if len(sys.argv) != 2 and len(sys.argv) != 4:
    usage()
    return
  elif sys.argv[1] not in ["train", "eval", "predict"]:
    usage()
    return

  mode = sys.argv[1]
  model_dir = ""
  BATCH_SIZE = 16
  predict_image_dir = ""

  if len(sys.argv) == 2:
    model_dir = "./tmp/color_batch"
    if mode == "train" or mode == "eval":
      BATCH_SIZE = 16
    if mode == "predict":
      predict_image_dir = "./predict/"

  if len(sys.argv) == 4:
    model_dir = sys.argv[2]
    if mode == "train" or mode == "eval":
      BATCH_SIZE = int(sys.argv[3])
    if mode == "predict":
      predict_image_dir = sys.argv[3]

  # Create the Estimator
  #cnn_model_fn color_cnn_fn
  color_painter = tf.estimator.Estimator(
      model_fn=color_cnn_fn, model_dir=model_dir + str(BATCH_SIZE))

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  # tensors_to_log = {"images": "output_rgb"}
  # logging_hook = tf.train.LoggingTensorHook(
  #     tensors=tensors_to_log, every_n_iter=100)

  # Train the model
  if mode == "train":
    print("Mode is train")
    color_painter.train(
        input_fn=dataset_input_fn,
        steps=STEPS,
        hooks=None)

  # Evaluate the model and print results
  if mode == "eval":
    print("Mode is eval")
    eval_results = color_painter.evaluate(input_fn=eval_input_fn)
    print(eval_results)

  # Predict by using the model and print results
  if mode == "predict":
    print("Mode is predict")
    print("predict_image_dir is: " + predict_image_dir)
    # predict_image_dir = "./predict/"
    predict_results = color_painter.predict(input_fn=eval_input_fn)
    print(type(predict_results))
    print(type(predict_results.next()))
    print(type(predict_results.next()["predict"]))
    print(predict_results.next()["predict"].shape)
    for i in range(1, 11):
      next_predict = predict_results.next()
      io.imsave(predict_image_dir + "predict"+ str(i) +".jpg", next_predict["predict"])
      io.imsave(predict_image_dir + "truth"+ str(i) +".jpg", next_predict["truth"])

if __name__ == "__main__":
  tf.app.run()