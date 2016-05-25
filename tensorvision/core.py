#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Core functions of TV."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf


def build_graph(hypes, modules, train=True):
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

    q, logits, decoder, = {}, {}, {}
    image_batch, label_batch = {}, {}
    eval_lists = {}

    if train:
        # Add Input Producers to the Graph
        with tf.name_scope('Input'):
            q['train'] = data_input.create_queues(hypes, 'train')
            input_batch = data_input.inputs(hypes, q['train'], 'train',
                                            hypes['dirs']['data_dir'])
            image_batch['train'], label_batch['train'] = input_batch

        logits['train'] = arch.inference(hypes, image_batch['train'], 'train')

        decoder['train'] = objective.decoder(hypes, logits['train'])

        # Add to the Graph the Ops for loss calculation.
        loss = objective.loss(hypes, decoder['train'], label_batch['train'])

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = solver.training(hypes, loss, global_step=global_step)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_lists['train'] = objective.evaluation(hypes, decoder['train'],
                                                   label_batch['train'])
    else:
        train_op = None
        loss = None

    # Validation Cycle to the Graph
    if train:
        scope_name = 'Validation'
    else:
        scope_name = 'Inference'
    with tf.name_scope(scope_name):
        with tf.name_scope('Input'):
            q['val'] = data_input.create_queues(hypes, 'val')
            input_batch = data_input.inputs(hypes, q['val'], 'val',
                                            hypes['dirs']['data_dir'])
            image_batch['val'], label_batch['val'] = input_batch

        if train:
            tf.get_variable_scope().reuse_variables()

        logits['val'] = arch.inference(hypes, image_batch['val'], 'val')

        decoder['val'] = objective.decoder(hypes, logits['val'])

        eval_lists['val'] = objective.evaluation(hypes, decoder['val'],
                                                 label_batch['val'])

    return q, train_op, loss, eval_lists


def start_tv_session(hypes):
    """
    Run one evaluation against the full epoch of data.

    Parameters
    ----------
    hypes : dict
        Hyperparameters

    Returns
    -------
    tuple
        (sess, saver, summary_op, summary_writer, threads)
    """
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
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(hypes['dirs']['output_dir'],
                                            graph=sess.graph)

    return sess, saver, summary_op, summary_writer, coord, threads


def do_eval(hypes, eval_list, phase, sess):
    """
    Run one evaluation against the full epoch of data.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    eval_list : list of tuples
        Each tuple should contain a string (name if the metric) and a
        tensor (storing the result of the metric).
    phase : str
        Describes the data the evaluation is run on.
    sess : tf.Session
        The session in which the model has been trained.

    Returns
    -------
    tuple of lists
        List of names and evaluation results
    """
    # And run one epoch of eval.
    # Checking for List for compability
    if type(eval_list[phase]) is list:
        eval_names, eval_op = zip(*eval_list[phase])

    else:
        logging.warning("Passing eval_op directly is deprecated. "
                        "Pass a list of tuples instead.")
        eval_names = ['Accuracy']
        eval_op = [eval_list[phase]]

    assert(len(eval_names) == len(eval_op))

    if phase == 'train':
        num_examples = hypes['data']['num_examples_per_epoch_for_train']
    if phase == 'val':
        num_examples = hypes['data']['num_examples_per_epoch_for_eval']

    steps_per_epoch = num_examples // hypes['solver']['batch_size']
    num_examples = steps_per_epoch * hypes['solver']['batch_size']

    logging.info('Data: % s  Num examples: % d ' % (phase, num_examples))
    # run evaluation on num_examples many images
    results = sess.run(eval_op)
    logging.debug('Output of eval: %s', results)
    for step in xrange(1, steps_per_epoch):
        results = map(np.add, results, sess.run(eval_op))

    avg_results = [result / steps_per_epoch for result in results]

    for name, value in zip(eval_names, avg_results):
        logging.info('%s : % 0.04f ' % (name, value))

    return eval_names, avg_results
