"""Trains, Evaluates and Saves the model network using a Queue."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os.path
import time
import logging
import sys
import imp
from shutil import copyfile

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import utils as utils

flags = tf.app.flags
FLAGS = flags.FLAGS


# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


def _copy_parameters_to_traindir(input_file, target_name, target_dir):
    """Helper to copy files defining the network to the saving dir.

    Args:
      input_file: name of source file
      target_name: target name
      traindir: directory where training data is saved
    """
    target_file = os.path.join(target_dir, target_name)
    copyfile(input_file, target_file)


def initialize_training_folder(hypes, train_dir):
    """Creating the training folder and copy all model files into it.

    The model will be executed from the training folder and all
    outputs will be saved there.

    Args:
      hypes: hypes
      train_dir: The training folder
    """
    target_dir = os.path.join(train_dir, "model_files")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Creating an additional logging saving the console outputs
    # into the training folder
    logging_file = os.path.join(train_dir, "output.log")
    filewriter = logging.FileHandler(logging_file, mode='w')
    formatter = logging.Formatter(
        '%(asctime)s %(name)-3s %(levelname)-3s %(message)s')
    filewriter.setLevel(logging.INFO)
    filewriter.setFormatter(formatter)
    logging.getLogger('').addHandler(filewriter)

    # TODO: read more about loggers and make file logging neater.

    config_file = tf.app.flags.FLAGS.config
    _copy_parameters_to_traindir(config_file, "hypes.json", target_dir)
    _copy_parameters_to_traindir(
        hypes['model']['input_file'], "data_input.py", target_dir)
    _copy_parameters_to_traindir(
        hypes['model']['arch_file'], "architecture.py", target_dir)
    _copy_parameters_to_traindir(
        hypes['model']['solver_file'], "solver.py", target_dir)


def maybe_download_and_extract(hypes, train_dir):
    target_dir = os.path.join(train_dir, "model_files")
    data_input = imp.load_source("input", hypes['model']['input_file'])
    data_input.maybe_download_and_extract(hypes, utils.cfg.data_dir)


def write_precision_to_summary(precision, summary_writer, name, global_step, sess):
    # write result to summary
    summary = tf.Summary()
    # summary.ParseFromString(sess.run(summary_op))
    summary.value.add(tag='Evaluation/' + name + ' Precision',
                      simple_value=precision)
    summary_writer.add_summary(summary, global_step)


def run_training(hypes, train_dir):
    """Train model for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test on MNIST.

    # Tell TensorFlow that the model will be built into the default Graph.
    target_dir = os.path.join(train_dir, "model_files")
    data_input = imp.load_source("input", hypes['model']['input_file'])
    arch = imp.load_source("arch", hypes['model']['arch_file'])
    solver = imp.load_source("solver", hypes['model']['solver_file'])

    with tf.Graph().as_default():

        global_step = tf.Variable(0.0, trainable=False)

        # TODO: fix/train-val

        with tf.name_scope('Input'):
            image_batch, label_batch = data_input.distorted_inputs(
                hypes, utils.cfg.data_dir)

        # Build a Graph that computes predictions from the inference arch.
        logits = arch.inference(hypes, image_batch, train=True)

        # Build Graph for Validation. This Graph shares Variabels with
        # the training Graph
        with tf.name_scope('Validation'):
            with tf.name_scope('Input_train_data'):
                image_batch_val, label_batch_val = data_input.distorted_inputs(hypes,
                                                                               utils.cfg.data_dir)
            with tf.name_scope('Input_val_data'):
                image_batch_train, label_batch_train = data_input.inputs(hypes, False,
                                                                         utils.cfg.data_dir)
            with tf.name_scope('Input_test_data'):
                image_batch_test, label_batch_test = data_input.inputs(hypes, True,
                                                                       utils.cfg.data_dir)

            # activate the reuse of Variabels
            tf.get_variable_scope().reuse_variables()

            # Build arch for Validation and Evaluation Data
            logits_train = arch.inference(hypes, image_batch_train, train=False)
            logits_val = arch.inference(hypes, image_batch_val, train=False)
            logits_test = arch.inference(hypes, image_batch_test, train=False)

        # Add to the Graph the Ops for loss calculation.
        loss = arch.loss(hypes, logits, label_batch)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = solver.training(hypes, loss, global_step=global_step)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_train = arch.evaluation(hypes, logits_train, label_batch_train)
        eval_val = arch.evaluation(hypes, logits_val, label_batch_val)
        eval_test = arch.evaluation(hypes, logits_test, label_batch_test)
        eval_correct = arch.evaluation(hypes, logits, label_batch)

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
        for step in xrange(hypes['solver']['max_steps']):
            start_time = time.time()

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss])

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                duration = time.time() - start_time
                examples_per_sec = hypes['solver']['batch_size'] / duration
                sec_per_batch = float(duration)
                logging.info('Step %d: loss = %.2f ( %.3f sec (per Batch); %.1f examples/sec;)'
                             % (step, loss_value,
                                sec_per_batch, examples_per_sec))
                # Update the events file.
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save a checkpoint and evaluate the model periodically.
            if (step+1) % 1000 == 0 or (step + 1) == hypes['solver']['max_steps']:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                # Evaluate against the training set.

            if (step+1) % 1000 == 0 or (step + 1) == hypes['solver']['max_steps']:
                logging.info(
                    'Doing Evaluate with whole epoche of Training Data:')
                precision = utils.do_eval(sess,
                                          eval_train,
                                          hypes['data'][
                                              'num_examples_per_epoch_for_train'],
                                          hypes,
                                          name="Train")
                write_precision_to_summary(
                    precision, summary_writer, "Train", step, sess)

                #logging.info('Validation Data Eval:')
                # TODO: Analyse Validation Error.
                # precision= utils.do_eval(sess,
                #                         eval_val,
                #                         hypes.num_examples_per_epoch_for_train,
                #                         hypes,
                #                         name="Val")
                #write_precision_to_summary(precision, summary_writer,"Val" , step, sess)

                logging.info('Doing Evaluation with Testing Data')
                precision = utils.do_eval(sess,
                                          eval_test,
                                          hypes['data'][
                                              'num_examples_per_epoch_for_eval'],
                                          hypes,
                                          name="Test")
                write_precision_to_summary(
                    precision, summary_writer, "Test", step, sess)


def main(_):
    if FLAGS.config == "example_params.py":
        logging.info("Training on default config.")
        logging.info(
            "Use training.py --config=your_config.py to train different models")

    with open(tf.app.flags.FLAGS.config, 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)

    train_dir = utils.get_train_dir()
    initialize_training_folder(hypes, train_dir)
    maybe_download_and_extract(hypes, train_dir)
    run_training(hypes, train_dir)


if __name__ == '__main__':
    tf.app.run()
