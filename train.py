"""Trains, Evaluates and Saves the model network using a Queue."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import logging
import sys
import numpy
import imp
from shutil import copyfile

import tensorflow.python.platform
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

 
import utils as utils



logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

flags = tf.app.flags
FLAGS = flags.FLAGS



def _copy_parameters_to_traindir(input_file, target_name, target_dir):
  """Helper to copy files defining the network to the saving dir.

  Args:
    input_file: name of source file
    target_name: target name
    traindir: directory where training data is saved
  """
  target_file = os.path.join(target_dir, target_name)
  copyfile(input_file, target_file)
    
    


def initialize_training_folder(train_dir):
  target_dir = os.path.join(train_dir, "model_files")  
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)
  config_file = tf.app.flags.FLAGS.config
  params = imp.load_source("params", config_file)  
  _copy_parameters_to_traindir(config_file, "params.py", target_dir)
  _copy_parameters_to_traindir(params.input_file, "input.py", target_dir)
  _copy_parameters_to_traindir(params.network_file, "network.py", target_dir)
  _copy_parameters_to_traindir(params.opt_file, "optimizer.py", target_dir)


def maybe_download_and_extract(train_dir):
  target_dir = os.path.join(train_dir, "model_files")  
  data_input = imp.load_source("input", os.path.join(target_dir, "input.py"))
  data_input.maybe_download_and_extract(utils.cfg.data_dir)


def run_training(train_dir):
  """Train model for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.

  # Tell TensorFlow that the model will be built into the default Graph.


  target_dir = os.path.join(train_dir, "model_files")  
  data_input = imp.load_source("input", os.path.join(target_dir, "input.py"))
  network = imp.load_source("network", os.path.join(target_dir, "network.py"))
  opt = imp.load_source("objective", os.path.join(target_dir, "optimizer.py"))
  params = imp.load_source("params", os.path.join(target_dir, "params.py"))

  with tf.Graph().as_default():

    global_step = tf.Variable(0.0, trainable=False)

    with tf.name_scope('Input'):
      image_batch, label_batch = data_input.distorted_inputs(utils.cfg.data_dir,
                                                             params.batch_size)
    # Generate placeholders for the images and labels.
      keep_prob = utils.placeholder_inputs(params.batch_size)

    # Build a Graph that computes predictions from the inference network.
    logits = network.inference(image_batch, keep_prob)

    # Add to the Graph the Ops for loss calculation.
    loss = network.loss(logits, label_batch)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = opt.training(loss, global_step=global_step,
                            learning_rate=params.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = network.evaluation(logits, label_batch)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(train_dir,
                                            graph_def=sess.graph_def)

    # And then after everything is built, start the training loop.
    for step in xrange(params.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = utils.fill_feed_dict(keep_prob,
                                       train = True)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        duration = time.time() - start_time
        examples_per_sec = params.batch_size / duration
        sec_per_batch = float(duration)
        print('Step %d: loss = %.2f ( %.3f sec (per Batch); %.1f examples/sec;)'
                                     % (step, loss_value,
                                     sec_per_batch, examples_per_sec))
        # Update the events file.
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == params.max_steps:
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path , global_step=step)
        # Evaluate against the training set.

      if (step + 1) % 5000 == 0 or (step + 1) == params.max_steps:  
        print('Training Data Eval:')
        utils.do_eval(sess,
                      eval_correct,
                      keep_prob,
                      params.num_examples_per_epoch_for_train,
                      params)

def main(_):
  if FLAGS.config == "example_params.py":
    logging.info("Training on default config.")
    logging.info("Use training.py --config=your_config.py to train different models")

  train_dir = utils.get_train_dir()
  initialize_training_folder(train_dir)
  maybe_download_and_extract(train_dir)
  run_training(train_dir)



if __name__ == '__main__':
  tf.app.run()
