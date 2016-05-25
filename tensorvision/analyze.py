#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluates the model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
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

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import tensorflow as tf

import tensorvision.utils as utils
import tensorvision.core as core

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('logdir', None,
                    'Directory where logs are stored.')


def _load_weights(checkpoint_dir, sess, saver):
    """
    Load the weights of a model stored in saver.

    Parameters
    ----------
    checkpoint_dir : str
        The directory of checkpoints.
    sess : tf.Session
        A Session to use to restore the parameters.
    saver : tf.train.Saver
    """
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        logging.info(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)


def do_analyze(logdir):
    """
    Analyze a trained model.

    This will load model files and weights found in logdir and run a basic
    analysis.

    Parameters
    ----------
    logdir : string
        Directory with logs.
    """
    hypes = utils.load_hypes_from_logdir(logdir)
    modules = utils.load_modules_from_logdir(logdir)
    data_input, arch, objective, solver = modules

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():

        # build the graph based on the loaded modules
        graph_ops = core.build_graph(hypes, modules, train=False)
        q, train_op, loss, eval_lists = graph_ops
        q = graph_ops[0]

        # prepaire the tv session
        sess_coll = core.start_tv_session(hypes)
        sess, saver, summary_op, summary_writer, coord, threads = sess_coll

        _load_weights(logdir, sess, saver)
        # Start the data load
        data_input.start_enqueuing_threads(hypes, q['val'], 'val', sess,
                                           hypes['dirs']['data_dir'])

    return core.do_eval(hypes, eval_lists, 'val', sess)


def main(_):
    """Run main function."""
    if FLAGS.logdir is None:
        logging.error("No logdir are given.")
        logging.error("Usage: tv-analyze --logdir dir")
        exit(1)

    if FLAGS.gpus is None:
        if 'TV_USE_GPUS' in os.environ:
            if os.environ['TV_USE_GPUS'] == 'force':
                logging.error('Please specify a GPU.')
                logging.error('Usage tv-train --gpus <ids>')
                exit(1)
            else:
                gpus = os.environ['TV_USE_GPUS']
                logging.info("GPUs are set to: %s", gpus)
                os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    else:
        logging.info("GPUs are set to: %s", FLAGS.gpus)
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus

    utils.load_plugins()

    logging.info("Starting to analyze model in '%s'", FLAGS.logdir)
    do_analyze(FLAGS.logdir)


if __name__ == '__main__':
    tf.app.run()
