#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the model network using a queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imp
import json
import logging
import numpy as np
import os.path
import sys

# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)


import time

from shutil import copyfile

from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

import tensorvision.utils as utils
import tensorvision.core as core

flags = tf.app.flags
FLAGS = flags.FLAGS


def _copy_parameters_to_traindir(hypes, input_file, target_name, target_dir):
    """
    Helper to copy files defining the network to the saving dir.

    Parameters
    ----------
    input_file : str
        name of source file
    target_name : str
        target name
    traindir : str
        directory where training data is saved
    """
    target_file = os.path.join(target_dir, target_name)
    input_file = os.path.os.path.realpath(
        os.path.join(hypes['dirs']['base_path'], input_file))
    copyfile(input_file, target_file)


def _start_enqueuing_threads(hypes, q, sess, data_input):
    """
    Start the enqueuing threads of the data_input module.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    sess : session
    q : queue
    data_input: data_input
    """
    with tf.name_scope('data_load'):
            data_input.start_enqueuing_threads(hypes, q['train'], 'train',
                                               sess, hypes['dirs']['data_dir'])
            data_input.start_enqueuing_threads(hypes, q['val'], 'val', sess,
                                               hypes['dirs']['data_dir'])


def initialize_training_folder(hypes):
    """
    Creating the training folder and copy all model files into it.

    The model will be executed from the training folder and all
    outputs will be saved there.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    """
    target_dir = os.path.join(hypes['dirs']['output_dir'], "model_files")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Creating an additional logging saving the console outputs
    # into the training folder
    logging_file = os.path.join(hypes['dirs']['output_dir'], "output.log")
    utils.create_filewrite_handler(logging_file)

    # TODO: read more about loggers and make file logging neater.

    target_file = os.path.join(target_dir, 'hypes.json')
    with open(target_file, 'w') as outfile:
        json.dump(hypes, outfile, indent=2)
    _copy_parameters_to_traindir(
        hypes, hypes['model']['input_file'], "data_input.py", target_dir)
    _copy_parameters_to_traindir(
        hypes, hypes['model']['architecture_file'], "architecture.py",
        target_dir)
    _copy_parameters_to_traindir(
        hypes, hypes['model']['objective_file'], "objective.py", target_dir)
    _copy_parameters_to_traindir(
        hypes, hypes['model']['optimizer_file'], "solver.py", target_dir)


def build_training_graph(hypes, modules):
    """
    Build the tensorflow graph out of the model files.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    modules : tuple
        The modules load in utils.

    Returns
    -------
    tuple
        (q, train_op, loss, eval_lists) where
        q is a dict with keys 'train' and 'val' which includes queues,
        train_op is a tensorflow op,
        loss is a float,
        eval_lists is a dict with keys 'train' and 'val'
    """
    data_input, arch, objective, solver = modules

    global_step = tf.Variable(0.0, trainable=False)
    learning_rate = tf.placeholder(tf.float32)
    tf.scalar_summary('learning_rate', learning_rate)

    q, logits, decoder, = {}, {}, {}
    image_batch, label_batch = {}, {}
    eval_lists = {}

    # Add Input Producers to the Graph
    with tf.name_scope('Input'):
        q['train'] = data_input.create_queues(hypes, 'train')
        input_batch = data_input.inputs(hypes, q['train'], 'train',
                                        hypes['dirs']['data_dir'])
        image_batch['train'], label_batch['train'] = input_batch

    logits['train'] = arch.inference(hypes, image_batch['train'],
                                     train=True)

    decoder['train'] = objective.decoder(hypes, logits['train'])

    # Add to the Graph the Ops for loss calculation.
    loss = objective.loss(hypes, decoder['train'], label_batch['train'])

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = solver.training(hypes, loss,
                               global_step=global_step,
                               learning_rate=learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    if hasattr(objective, 'evaluation'):
        eval_lists['train'] = objective.evaluation(hypes, decoder['train'],
                                                   label_batch['train'])

    # Validation Cycle to the Graph
    with tf.name_scope('Validation'):
        with tf.name_scope('Input'):
            q['val'] = data_input.create_queues(hypes, 'val')
            input_batch = data_input.inputs(hypes, q['val'], 'val',
                                            hypes['dirs']['data_dir'])
            image_batch['val'], label_batch['val'] = input_batch

            tf.get_variable_scope().reuse_variables()

        logits['val'] = arch.inference(hypes, image_batch['val'],
                                       train=False)

        decoder['val'] = objective.decoder(hypes, logits['val'])

        if hasattr(objective, 'evaluation'):
            eval_lists['val'] = objective.evaluation(hypes, decoder['val'],
                                                     label_batch['val'])

    return q, train_op, loss, eval_lists, learning_rate


def maybe_download_and_extract(hypes):
    """
    Download the data if it isn't downloaded by now.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    """
    f = os.path.join(hypes['dirs']['base_path'], hypes['model']['input_file'])
    data_input = imp.load_source("input", f)
    if hasattr(data_input, 'maybe_download_and_extract'):
        data_input.maybe_download_and_extract(hypes, hypes['dirs']['data_dir'])


def _write_evaluation_to_summary(evaluation_results, summary_writer, phase,
                                 global_step, sess):
    """
    Write the evaluation_results to the summary file.

    Parameters
    ----------
    evaluation_results : tuple
        The output of do_eval
    summary_writer : tf.train.SummaryWriter
    phase : string
        Name of Operation to write
    global_step : tensor or int
        Xurrent training step
    sess : tf.Session
    """
    # write result to summary
    summary = tf.Summary()
    eval_names, avg_results = evaluation_results
    for name, result in zip(eval_names, avg_results):
        summary.value.add(tag='Evaluation/' + phase + '/' + name,
                          simple_value=result)
    summary_writer.add_summary(summary, global_step)


def _print_training_status(hypes, step, loss_value, summary_str,
                           start_time, sess_coll):
    duration = (time.time() - start_time) / int(utils.cfg.step_show)
    examples_per_sec = hypes['solver']['batch_size'] / duration
    sec_per_batch = float(duration)
    info_str = utils.cfg.step_str

    sess, saver, summary_op, summary_writer, coord, threads = sess_coll

    # Update the events file.
    summary_writer.add_summary(summary_str, step)

    logging.info(info_str.format(step=step,
                                 total_steps=hypes['solver']['max_steps'],
                                 loss_value=loss_value,
                                 sec_per_batch=sec_per_batch,
                                 examples_per_sec=examples_per_sec)
                 )


def _write_checkpoint_to_disk(hypes, step, sess_coll):
    sess, saver, summary_op, summary_writer, coord, threads = sess_coll
    checkpoint_path = os.path.join(hypes['dirs']['output_dir'],
                                   'model.ckpt')
    saver.save(sess, checkpoint_path, global_step=step)


def _do_evaluation(hypes, step, sess_coll, eval_dict):
    sess, saver, summary_op, summary_writer, coord, threads = sess_coll
    logging.info('Doing Evaluate with Training Data.')

    eval_results = core.do_eval(hypes, eval_dict, phase='train',
                                sess=sess)
    _write_evaluation_to_summary(eval_results, summary_writer,
                                 "Train", step, sess)

    logging.info('Doing Evaluation with Testing Data.')
    eval_results = core.do_eval(hypes, eval_dict, phase='val',
                                sess=sess)
    _write_evaluation_to_summary(eval_results, summary_writer,
                                 'val', step, sess)


def _write_eval_dict_to_summary(eval_dict, summary_writer, global_step):
    summary = tf.Summary()
    for name, result in eval_dict:
        summary.value.add(tag='Evaluation/' + 'python' + '/' + name,
                          simple_value=result)
    summary_writer.add_summary(summary, global_step)
    return


def _write_images_to_summary(images, summary_writer, step):
    for name, image in images:
        image = image.astype('float32')
        shape = image.shape
        image = image.reshape(1, shape[0], shape[1], shape[2])
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                log_image = tf.image_summary(name, image)
            with tf.Session() as sess:
                summary_str = sess.run([log_image])
                summary_writer.add_summary(summary_str[0], step)
    return


def _do_python_evaluation(hypes, step, sess_coll, objective,
                          image_pl, softmax):
    logging.info('Doing Python Evaluation.')
    sess, saver, summary_op, summary_writer, coord, threads = sess_coll
    eval_dict, images = objective.evaluate(hypes, sess, image_pl, softmax)

    utils.print_eval_dict(eval_dict)
    _write_eval_dict_to_summary(eval_dict, summary_writer, step)
    _write_images_to_summary(images, summary_writer, step)

    return


def run_training_step(hypes, step, start_time, graph_ops, sess_coll,
                      modules, image_pl, softmax):
    """Run one iteration of training."""
    # Unpack operations for later use
    sess, saver, summary_op, summary_writer, coord, threads = sess_coll

    q, train_op, loss, eval_dict, learning_rate = graph_ops
    data_input, arch, objective, solver = modules

    lr = solver.get_learning_rate(hypes, step)
    feed_dict = {learning_rate: lr}

    # Run the training Step

    if step % int(utils.cfg.step_show):
        sess.run([train_op], feed_dict=feed_dict)

    # Write the summaries and print an overview fairly often.
    elif step % int(utils.cfg.step_show) == 0:
        # Print status to stdout.
        _, loss_value, summary_str = sess.run([train_op, loss, summary_op],
                                              feed_dict=feed_dict)
        _print_training_status(hypes, step, loss_value, summary_str,
                               start_time, sess_coll)
        # Reset timer
        start_time = time.time()

    # Do a evaluation and print the current state
    if (step + 1) % int(utils.cfg.step_eval) == 0 or \
       (step + 1) == hypes['solver']['max_steps']:
        # write checkpoint to disk
        if hasattr(objective, 'evaluate'):
            _do_python_evaluation(hypes, step, sess_coll, objective,
                                  image_pl, softmax)
        if hasattr(objective, 'evaluation'):
            logging.warning("Defining evaluation Tensors is depricated.")
            logging.warning("This might be removed in future Versions.")
            _do_evaluation(hypes, step, sess_coll, eval_dict)
        # Reset timer
        start_time = time.time()

    # Save a checkpoint periodically.
    if (step + 1) % int(utils.cfg.step_write) == 0 or \
       (step + 1) == hypes['solver']['max_steps']:
        # write checkpoint to disk
        _write_checkpoint_to_disk(hypes, step, sess_coll)
        # Reset timer
        start_time = time.time()

    return start_time


def _create_input_placeholder():
    image_pl = tf.placeholder(tf.float32)
    label_pl = tf.placeholder(tf.float32)
    return image_pl, label_pl


def do_training(hypes):
    """
    Train model for a number of steps.

    This trains the model for at most hypes['solver']['max_steps'].
    It shows an update every utils.cfg.step_show steps and writes
    the model to hypes['dirs']['output_dir'] every utils.cfg.step_eval
    steps.

    Paramters
    ---------
    hypes : dict
        Hyperparameters
    """
    # Get the sets of images and labels for training, validation, and
    # test on MNIST.

    modules = utils.load_modules_from_hypes(hypes)
    data_input, arch, objective, solver = modules

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():

        # build the graph based on the loaded modules
        graph_ops = build_training_graph(hypes, modules)
        q = graph_ops[0]

        # prepaire the tv session
        sess_coll = core.start_tv_session(hypes)
        sess, saver, summary_op, summary_writer, coord, threads = sess_coll

        with tf.name_scope('Validation'):
            image_pl, label_pl = _create_input_placeholder()
            image = tf.expand_dims(image_pl, 0)
            softmax = core.build_inference_graph(hypes, modules,
                                                 image=image,
                                                 label=label_pl)

        # Start the data load
        _start_enqueuing_threads(hypes, q, sess, data_input)

        # And then after everything is built, start the training loop.
        start_time = time.time()
        for step in xrange(hypes['solver']['max_steps']):
            start_time = run_training_step(hypes, step, start_time,
                                           graph_ops, sess_coll, modules,
                                           image_pl, softmax)
            if hasattr(solver, 'update_learning_rate'):
                solver.update_learning_rate(hypes, step)

        # stopping input Threads
        coord.request_stop()
        coord.join(threads)


def continue_training(logdir):
    """
    Continues training of a model.

    This will load model files and weights found in logdir and continues
    an aborted training.

    Parameters
    ----------
    logdir : string
        Directory with logs.
    """
    hypes = utils.load_hypes_from_logdir(logdir)
    modules = utils.load_modules_from_logdir(logdir)
    data_input, arch, objective, solver = modules

    # append output to output.log
    logging_file = os.path.join(logdir, 'output.log')
    utils.create_filewrite_handler(logging_file, mode='a')

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default() as graph:

        # build the graph based on the loaded modules
        graph_ops = build_training_graph(hypes, modules)
        q = graph_ops[0]

        # prepaire the tv session
        sess_coll = core.start_tv_session(hypes)
        sess, saver, summary_op, summary_writer, coord, threads = sess_coll

        if hasattr(objective, 'evaluate'):
            with tf.name_scope('Validation'):
                image_pl, label_pl = _create_input_placeholder()
                image = tf.expand_dims(image_pl, 0)
                softmax = core.build_inference_graph(hypes, modules,
                                                     image=image,
                                                     label=label_pl)

        # Load weights from logdir
        cur_step = core.load_weights(logdir, sess, saver)

        # Start the data load
        _start_enqueuing_threads(hypes, q, sess, data_input)

        # And then after everything is built, start the training loop.
        start_time = time.time()
        for step in xrange(cur_step+1, hypes['solver']['max_steps']):
            start_time = run_training_step(hypes, step, start_time,
                                           graph_ops, sess_coll, modules,
                                           image_pl, softmax)

        # stopping input Threads
        coord.request_stop()
        coord.join(threads)


def main(_):
    """Run main function."""
    if FLAGS.hypes is None:
        logging.error("No hypes are given.")
        logging.error("Usage: tv-train --hypes hypes.json")
        exit(1)

    with open(tf.app.flags.FLAGS.hypes, 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)

    utils.set_gpus_to_use()
    utils.load_plugins()
    utils.set_dirs(hypes, tf.app.flags.FLAGS.hypes)

    logging.info("Initialize training folder")
    initialize_training_folder(hypes)
    maybe_download_and_extract(hypes)
    logging.info("Start training")
    do_training(hypes)


if __name__ == '__main__':
    tf.app.run()
