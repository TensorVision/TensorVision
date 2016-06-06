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
        file = os.path.basename(ckpt.model_checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_dir, file)
        saver.restore(sess, checkpoint_path)


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


# Utility functions for analyzing models
def get_confusion_matrix(correct_seg, segmentation, elements=None):
    """
    Get the confuscation matrix of a segmentation image and its ground truth.

    The confuscation matrix is a detailed count of which classes i were
    classifed as classes j, where i and j take all (elements) names.

    Parameters
    ----------
    correct_seg : numpy array
        Representing the ground truth.
    segmentation : numpy array
        Predicted segmentation
    elements : iterable
        A list / set or another iterable which contains the possible
        segmentation classes (commonly 0 and 1).

    Returns
    -------
    dict
        A confusion matrix m[correct][classified] = number of pixels in this
        category.
    """
    height, width = correct_seg.shape

    # Get classes
    if elements is None:
        elements = set(np.unique(correct_seg))
        elements = elements.union(set(np.unique(segmentation)))
        logging.debug("elements parameter not given to get_confusion_matrix")
        logging.debug("  assume '%s'", elements)

    # Initialize confusion matrix
    confusion_matrix = {}
    for i in elements:
        confusion_matrix[i] = {}
        for j in elements:
            confusion_matrix[i][j] = 0

    for x in range(width):
        for y in range(height):
            confusion_matrix[correct_seg[y][x]][segmentation[y][x]] += 1
    return confusion_matrix


def get_accuracy(n):
    r"""
    Get the accuracy from a confusion matrix n.

    The mean accuracy is calculated as

    .. math::

        t_i &= \sum_{j=1}^k n_{ij}\\
        acc(n) &= \frac{\sum_{i=1}^k n_{ii}}{\sum_{i=1}^k n_{ii}}

    Parameters
    ----------
    n : dict
        Confusion matrix which has integer keys 0, ..., nb_classes - 1;
        an entry n[i][j] is the count how often class i was classified as
        class j.

    Returns
    -------
    float
        accuracy (in [0, 1])

    References
    ----------
    .. [1] Martin Thoma (2016): A Survey of Semantic Segmentation,
       http://arxiv.org/abs/1602.06541

    Examples
    --------
    >>> n = {0: {0: 10, 1: 2}, 1: {0: 5, 1: 83}}
    >>> get_accuracy(n)
    0.93
    """
    return (float(n[0][0] + n[1][1]) /
            (n[0][0] + n[1][1] + n[0][1] + n[1][0]))


def get_mean_accuracy(n):
    """
    Get the mean accuracy from a confusion matrix n.

    Parameters
    ----------
    n : dict
        Confusion matrix which has integer keys 0, ..., nb_classes - 1;
        an entry n[i][j] is the count how often class i was classified as
        class j.

    Returns
    -------
    float
        mean accuracy (in [0, 1])

    Examples
    --------
    >>> n = {0: {0: 10, 1: 2}, 1: {0: 5, 1: 83}}
    >>> get_mean_accuracy(n)
    0.8882575757575758
    """
    t = []
    k = len(n[0])
    for i in range(k):
        t.append(sum([n[i][j] for j in range(k)]))
    return (1.0 / k) * sum([float(n[i][i]) / t[i] for i in range(k)])


def get_mean_iou(n):
    """
    Get mean intersection over union from a confusion matrix n.

    Parameters
    ----------
    n : dict
        Confusion matrix which has integer keys 0, ..., nb_classes - 1;
        an entry n[i][j] is the count how often class i was classified as
        class j.

    Returns
    -------
    float
        mean intersection over union (in [0, 1])

    Examples
    --------
    >>> n = {0: {0: 10, 1: 2}, 1: {0: 5, 1: 83}}
    >>> get_mean_iou(n)
    0.7552287581699346
    """
    t = []
    k = len(n[0])
    for i in range(k):
        t.append(sum([n[i][j] for j in range(k)]))
    return (1.0 / k) * sum([float(n[i][i]) / (t[i] - n[i][i] +
                            sum([n[j][i] for j in range(k)]))
                            for i in range(k)])


def get_frequency_weighted_iou(n):
    """
    Get frequency weighted intersection over union.

    Parameters
    ----------
    n : dict
        Confusion matrix which has integer keys 0, ..., nb_classes - 1;
        an entry n[i][j] is the count how often class i was classified as
        class j.

    Returns
    -------
    float
        frequency weighted iou (in [0, 1])

    Examples
    --------
    >>> n = {0: {0: 10, 1: 2}, 1: {0: 5, 1: 83}}
    >>> get_frequency_weighted_iou(n)
    0.8821437908496732
    """
    t = []
    k = len(n[0])
    for i in range(k):
        t.append(sum([n[i][j] for j in range(k)]))
    a = sum(t)**(-1)
    b = sum([(t[i] * n[i][i]) /
             (t[i] - n[i][i] + sum([n[j][i] for j in range(k)]))
             for i in range(k)])
    return a * b


def merge_cms(cm1, cm2):
    """
    Merge two confusion matrices.

    Parameters
    ----------
    cm1 : dict
        Confusion matrix which has integer keys 0, ..., nb_classes - 1;
        an entry cm1[i][j] is the count how often class i was classified as
        class j.
    cm2 : dict
        Another confusion matrix.

    Returns
    -------
    dict
        merged confusion matrix

    Examples
    --------
    >>> cm1 = {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
    >>> cm2 = {0: {0: 5, 1: 6}, 1: {0: 7, 1: 8}}
    >>> merge_cms(cm1, cm2)
    {0: {0: 6, 1: 8}, 1: {0: 10, 1: 12}}
    """
    assert 0 in cm1
    assert len(cm1[0]) == len(cm2[0])

    cm = {}
    k = len(cm1[0])
    for i in range(k):
        cm[i] = {}
        for j in range(k):
            cm[i][j] = cm1[i][j] + cm2[i][j]

    return cm


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
