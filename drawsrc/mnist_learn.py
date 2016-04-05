# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python mnist_learn.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import png
import sys
import random

from tensorflow.models.rnn.ptb import reader
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS


class DRAWModel(object):
  """The DRAW model."""

  def __init__(self, is_training, config, reuse=False):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size

    self._input_data = tf.placeholder(tf.float32, [batch_size, 784])
    self._targets = tf.placeholder(tf.float32, [batch_size, 784])


    # The MNIST digit
    inputs = tf.placeholder("float", [None, 784])

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    with tf.variable_scope('encode', reuse=reuse):
      enc_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
      if is_training and config.keep_prob < 1:
        enc_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(enc_lstm_cell, output_keep_prob=config.keep_prob)

    # Output matrix weight
    mean_W = tf.Variable(tf.random_normal([size, size], stddev=0.01))
    mean_bias = tf.Variable(tf.random_normal([size], stddev=0.01))
    stddev_W = tf.Variable(tf.random_normal([size, size], stddev=0.01))
    stddev_bias = tf.Variable(tf.random_normal([size], stddev=0.01))


    with tf.variable_scope('decode', reuse=reuse):
      dec_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
      if is_training and config.keep_prob < 1:
        dec_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(dec_lstm_cell, output_keep_prob=config.keep_prob)

    # Output matrix weight
    output_W = tf.Variable(tf.random_normal([size, 784], stddev=0.01))
    output_bias = tf.Variable(tf.random_normal([784], stddev=0.01))


    canvas = tf.placeholder("float", [None, 784])
    self._enc_state = enc_lstm_cell.zero_state(batch_size, tf.float32)
    self._dec_state = dec_lstm_cell.zero_state(batch_size, tf.float32)
    self._canvas_state = tf.zeros([batch_size, 784])


    # Wire together an unrolled RNN
    means=[]
    stddevs=[]
    outputs = []
    enc_state = self._enc_state
    dec_state = self._dec_state
    canvas_state = self._canvas_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        input_error = tf.sub(tf.to_float(self._input_data), tf.sigmoid(canvas_state))

        with tf.variable_scope('encode', reuse=reuse):
          lstm_input = tf.reshape(tf.concat(1,[tf.to_float(self._input_data),input_error]), [-1,784+784])
          (enc_cell_output, enc_state) = enc_lstm_cell(lstm_input, enc_state)
        sample_mean=tf.add(tf.matmul(enc_cell_output,mean_W), mean_bias)
        sample_stddev=tf.add(tf.matmul(enc_cell_output,stddev_W),stddev_bias)
        means.append(tf.reduce_sum(tf.mul(sample_mean,sample_mean), 1))
        stddevs.append(tf.reduce_sum(tf.mul(sample_stddev,sample_stddev), 1))
        sampled = tf.random_normal([batch_size, size], mean=sample_mean, stddev=sample_stddev)

        with tf.variable_scope('decode', reuse=reuse):
          (dec_cell_output, dec_state) = dec_lstm_cell(sampled, dec_state)
        canvas_state = tf.add(canvas_state, tf.add(tf.matmul(dec_cell_output, output_W), output_bias))
        outputs.append(tf.sigmoid(canvas_state))

    output = outputs[-1]

   # elementwise = tf.cast(tf.abs(tf.sub(self._targets, output)), dtype=tf.float64)
    elementwise = tf.cast(tf.select(tf.greater(tf.constant(0.5), self._targets),
                tf.sub(tf.ones([batch_size,784]), output), output), tf.float64)

    # Underflow and rounding errors may cause 0 to appear in elementwise, which causes inf cost and ultimately nans everywhere
    elementwise = tf.select(tf.equal(tf.constant(0, dtype=tf.float64), elementwise),
                            tf.cast(tf.fill([batch_size,784], sys.float_info.epsilon), tf.float64),
                            elementwise)

    reconstruction_loss = tf.neg(tf.log(tf.reduce_prod(elementwise, reduction_indices=[1])))

    means_tensor = tf.pack(means)
    stddevs_tensor = tf.pack(stddevs)
    latent_loss = tf.div(tf.sub(tf.reduce_sum(
      tf.sub(tf.add(means_tensor,
                    stddevs_tensor),
             tf.log(stddevs_tensor)), reduction_indices=[0])
      , tf.to_float(tf.constant(num_steps))), tf.to_float(tf.constant(2)))
    loss = tf.add(reconstruction_loss, tf.cast(latent_loss, tf.float64))
    self._debug_state = reconstruction_loss


    self._recon_loss = reconstruction_loss
    self._latent_loss = latent_loss
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_canvas_state = canvas_state
    self._outputs = tf.pack(outputs)

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def enc_state(self):
    return self._enc_state

  @property
  def dec_state(self):
    return self._dec_state

  @property
  def debug_state(self):
    return self._debug_state

  @property
  def canvas_state(self):
    return self._canvas_state

  @property
  def outputs(self):
    return self._outputs

  @property
  def recon_loss(self):
    return self._recon_loss

  @property
  def latent_loss(self):
    return self._latent_loss

  @property
  def cost(self):
    return self._cost

  @property
  def final_canvas_state(self):
    return self._final_canvas_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 0.5
  max_grad_norm = 1
  num_layers = 2
  num_steps = 10
  hidden_size = 16
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def scale(data):
  return np.maximum(0, np.minimum(255, 256*data))

def arrange_images(data):
  return scale(np.reshape(data, [-1, 28]))

def dump_images(input, outputs, x):
  f = open('/Users/mwagner/git/draw-network/imgs/%d.png' % x, 'wb')      # binary mode is important
  column_input = list(arrange_images(input))
  column_outputs = [arrange_images(x) for x in list(outputs)]

  merged = np.concatenate(column_outputs, axis=1)
  merged = np.concatenate([merged, np.full((np.shape(merged)[0],1), 255)], axis=1)
  merged = np.concatenate([merged, column_input], axis=1)
  print(np.max(merged))
  shape = np.shape(merged)
  w = png.Writer(shape[1], shape[0], greyscale=True)
  w.write(f, merged)
  f.close()

def run_epoch(session, m, data, eval_op, verbose=False, epoch=0):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // m.batch_size) - 1)
  start_time = time.time()
  costs = 0.0
  iters = 0
  enc_state = m.enc_state.eval()
  dec_state = m.dec_state.eval()
  canvas_state = m.canvas_state.eval()

  saver = tf.train.Saver()
  for step, x in enumerate(batch(data, m.batch_size)):
    cost, outputs,  _ = session.run([m.cost, m.outputs, eval_op],
                                 {m.input_data: x,
                                  m.targets: x,
                                  m.enc_state: enc_state,
                                  m.dec_state: dec_state,
                                  m.canvas_state: canvas_state})

    costs += cost
    iters += 1

    if verbose and step % 10 == 0:
      print("%.3f costs: %.3f current cost: %.3f speed: %.0f ips" %
            (step * 1.0 / epoch_size, costs / iters, cost,
             iters * m.batch_size / (time.time() - start_time)))
    if step % 100 == 0:
      global_step = epoch*epoch_size + iters
      print(np.shape(outputs))
      dump_images(x, outputs, global_step)
      save_path = saver.save(session, "/Users/mwagner/git/draw-network/checkpoints/lstm-only-bias", global_step=global_step)
      print("Model saved in file: %s" % save_path)


  return costs / iters


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)

def binarize(data):
  random = np.random.uniform(size=np.shape(data))
  index =np.greater(random,data)
  data[index] = 0
  data[np.logical_not(index)] = 1
  return data

def main(_):
  #TODO: binarize the input: http://homepages.inf.ed.ac.uk/imurray2/pub/08dbn_ais/dbn_ais.pdf
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  train_data, valid_data, test_data = binarize(mnist.train.images), binarize(mnist.validation.images), binarize(mnist.test.images)

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1

  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = DRAWModel(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = DRAWModel(is_training=False, config=config, reuse=True)
      mtest = DRAWModel(is_training=False, config=eval_config, reuse=True)

    tf.initialize_all_variables().run()
    for i in range(config.max_max_epoch):
      lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
      m.assign_lr(session, config.learning_rate * lr_decay)

      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      train_perplexity = run_epoch(session, m, train_data, m.train_op, epoch=i,
                                   verbose=True)
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
      valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
      print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))



    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
    print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
  tf.app.run()
