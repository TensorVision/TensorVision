from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datetime import datetime
import time
import os
import logging

import config as cfg


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('name', None,
                    'Folder where Data will be stored.')

#usage: train.py --config=my_model_params.py
flags.DEFINE_string('config', cfg.default_config,
                    'File storing model parameters.')

def get_train_dir():
  if FLAGS.name is None:
    train_dir = os.path.join(cfg.model_dir, cfg.default_name)
    logging.info("Saving/Loading Model from default Folder: %s ", train_dir)
    logging.info("Use --name=MYNAME to use Folder: %s ",
                 os.path.join(cfg.model_dir, "MYNAME"))
  else:
    train_dir = os.path.join(cfg.model_dir, FLAGS.name)

  return train_dir


def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
    keep_prob: keep_prob placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  
  keep_prob = tf.placeholder("float")
  return keep_prob


def fill_feed_dict(kb, train):
  """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    kb: The keep prob placeholder.
    train: whether data set is on train.

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.

  if train :
    feed_dict = {
        kb: 0.5}
  else :
      feed_dict = {
        kb: 1.0}
  return feed_dict



def do_eval(sess,
            eval_correct,
            keep_prob,
            num_examples,
            params):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    keep_prob: The keep prob placeholder.
  """
  # And run one epoch of eval.

  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch =  num_examples // params.batch_size
  num_examples = steps_per_epoch * params.batch_size
  feed_dict = fill_feed_dict(keep_prob,
                             train = False)
  for step in xrange(steps_per_epoch):
    start_time = time.time()
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
    duration = time.time() - start_time

    if step % 20 == 0:


      num_examples_per_step = params.batch_size
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = float(duration)

      format_str = ('%s: step %d,  %.3f sec (per batch); (%.1f examples/sec)')
      print (format_str % (datetime.now(), step, sec_per_batch,examples_per_sec))
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))