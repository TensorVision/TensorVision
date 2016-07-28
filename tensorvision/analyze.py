#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluates the model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os.path
import sys

import scipy as scp
import scipy.misc

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


def _create_input_placeholder():
    image_pl = tf.placeholder(tf.float32)
    label_pl = tf.placeholder(tf.float32)
    return image_pl, label_pl


def _write_images_to_logdir(images, logdir):
    logdir = os.path.join(logdir, "eval/")
    for name, image in images:
        save_file = os.path.join(logdir, name)
        scp.misc.imsave(save_file, image)


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

        # prepaire the tv session

        with tf.name_scope('Validation'):
            image_pl, label_pl = _create_input_placeholder()
            image = tf.expand_dims(image_pl, 0)
            softmax = core.build_inference_graph(hypes, modules,
                                                 image=image,
                                                 label=label_pl)

        sess_coll = core.start_tv_session(hypes)
        sess, saver, summary_op, summary_writer, coord, threads = sess_coll

        core.load_weights(logdir, sess, saver)

        eval_dict, images = objective.tensor_eval(hypes, sess, image_pl,
                                                  softmax)

        logging_file = os.path.join(logdir, "eval/analysis.log")
        utils.create_filewrite_handler(logging_file)

        utils.print_eval_dict(eval_dict)
        _write_images_to_logdir(images, logdir)
    return


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


def get_precision(n):
    """
    Get precision.

    Parameters
    ----------
    n : dict
        Confusion matrix which has integer keys 0, ..., nb_classes - 1;
        an entry n[i][j] is the count how often class i was classified as
        class j.

    Returns
    -------
    float
        precision (in [0, 1])

    Examples
    --------
    >>> n = {0: {0: 10, 1: 2}, 1: {0: 5, 1: 83}}
    >>> get_precision(n)
    0.9764705882352941
    """
    assert len(n) == 2, "Precision is only defined for binary problems"
    return float(n[1][1]) / (n[0][1] + n[1][1])


def get_recall(n):
    """
    Get recall.

    Parameters
    ----------
    n : dict
        Confusion matrix which has integer keys 0, ..., nb_classes - 1;
        an entry n[i][j] is the count how often class i was classified as
        class j.

    Returns
    -------
    float
        recall (in [0, 1])

    Examples
    --------
    >>> n = {0: {0: 10, 1: 2}, 1: {0: 5, 1: 83}}
    >>> get_recall(n)
    0.9431818181818182
    """
    assert len(n) == 2, "Recall is only defined for binary problems"
    return float(n[1][1]) / (n[1][0] + n[1][1])


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


def get_color_distribution(labeled_dataset):
    """
    Get the distribution of colors of masks in a labeled dataset.

    Parameters
    ----------
    labeled_dataset : list of dicts
        Each dict has to have the keys 'raw' and 'mask' which have the absolute
        path to image files.

    Returns
    -------
    dict
        Mapping colors to pixel counts.
    """
    colors = {}
    for item in labeled_dataset:
        im = scipy.misc.imread(item['mask'], flatten=False, mode='RGB')
        for y in range(im.shape[0]):
            for x in range(im.shape[1]):
                color = tuple(im[y][x])
                if color in colors:
                    colors[color] += 1
                else:
                    colors[color] = 1
    return colors


def get_class_distribution(hypes, labeled_dataset):
    """
    Get the distribution of classes in a labeled dataset.

    Parameters
    ----------
    hypes : dict
        The hyperparameters have to specify 'classes'.
    labeled_dataset : list of dicts
        Each dict has to have the keys 'raw' and 'mask' which have the absolute
        path to image files.

    Returns
    -------
    dict
        Mapping class indices according to hypes['classes'] to pixel counts.
    """
    classes = {}
    for item in labeled_dataset:
        im = utils.load_segmentation_mask(hypes, item['mask'])
        for y in range(im.shape[0]):
            for x in range(im.shape[1]):
                cl = im[y][x]
                if cl in classes:
                    classes[cl] += 1
                else:
                    classes[cl] = 1
    return classes


def main(_):
    """Run main function."""
    if FLAGS.logdir is None:
        logging.error("No logdir is given.")
        logging.error("Usage: tv-analyze --logdir dir")
        exit(1)

    utils.set_gpus_to_use()
    utils.load_plugins()

    logging.info("Starting to analyze model in '%s'", FLAGS.logdir)
    do_analyze(FLAGS.logdir)


if __name__ == '__main__':
    tf.app.run()
