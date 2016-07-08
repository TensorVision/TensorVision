"""Utils for TensorVision."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imp
import json
import logging
import os

from datetime import datetime
import matplotlib.cm as cm

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy
import sys

import tensorflow as tf

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('gpus', None,
                    ('Which gpus to use. For multiple GPUs use comma seperated'
                     'ids. [e.g. --gpus 0,3]'))


def set_dirs(hypes, hypes_fname):
    """
    Add directories to hypes.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    hypes_fname : str
        Path to hypes_file
    """
    if 'dirs' not in hypes:
        hypes['dirs'] = {}

    # Set base_path
    if 'base_path' not in hypes['dirs']:
        base_path = os.path.dirname(os.path.realpath(hypes_fname))
        hypes['dirs']['base_path'] = base_path
    else:
        base_path = hypes['dirs']['base_path']

    # Set output dir
    if 'output_dir' not in hypes['dirs']:
        if 'TV_DIR_RUNS' in os.environ:
            runs_dir = os.path.join(base_path, os.environ['TV_DIR_RUNS'])
        else:
            runs_dir = os.path.join(base_path, 'RUNS')

        # test for project dir
        if hasattr(FLAGS, 'project') and FLAGS.project is not None:
            runs_dir = os.path.join(runs_dir, FLAGS.project)

        if not FLAGS.save and FLAGS.name is None:
            output_dir = os.path.join(runs_dir, 'debug')
        else:
            json_name = hypes_fname.split('/')[-1].replace('.json', '')
            date = datetime.now().strftime('%Y_%m_%d_%H.%M')
            if FLAGS.name is not None:
                json_name = FLAGS.name + "_" + json_name
            run_name = '%s_%s' % (json_name, date)
            output_dir = os.path.join(runs_dir, run_name)

        hypes['dirs']['output_dir'] = output_dir

    # Set data dir
    if 'data_dir' not in hypes['dirs']:
        if 'TV_DIR_DATA' in os.environ:
            data_dir = os.path.join(base_path, os.environ['TV_DIR_DATA'])
        else:
            data_dir = os.path.join(base_path, 'DATA')

        hypes['dirs']['data_dir'] = data_dir

    return


def set_gpus_to_use():
    """Set the gpus to use."""
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


def load_modules_from_hypes(hypes):
    """Load all modules from the files specified in hypes.

    Namely the modules loaded are:
    input_file, architecture_file, objective_file, optimizer_file

    Parameters
    ----------
    hypes : dict
        Hyperparameters

    Returns
    -------
    hypes, data_input, arch, objective, solver
    """
    base_path = hypes['dirs']['base_path']
    _add_paths_to_sys(hypes)
    f = os.path.join(base_path, hypes['model']['input_file'])
    data_input = imp.load_source("input", f)
    f = os.path.join(base_path, hypes['model']['architecture_file'])
    arch = imp.load_source("arch", f)
    f = os.path.join(base_path, hypes['model']['objective_file'])
    objective = imp.load_source("objective", f)
    f = os.path.join(base_path, hypes['model']['optimizer_file'])
    solver = imp.load_source("solver", f)

    return data_input, arch, objective, solver


def _add_paths_to_sys(hypes):
    """
    Add all module dirs to syspath.

    This adds the dirname of all modules to path.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    """
    base_path = hypes['dirs']['base_path']
    for module in hypes['model'].values():
        path = os.path.realpath(os.path.join(base_path, module))
        sys.path.append(os.path.dirname(path))
    if 'path' in hypes:
            for path in hypes['path']:
                path = os.path.realpath(os.path.join(base_path, path))
                sys.path.append(path)
    return


def load_modules_from_logdir(logdir):
    """Load hypes from the logdir.

    Namely the modules loaded are:
    input_file, architecture_file, objective_file, optimizer_file

    Parameters
    ----------
    logdir : string
        Path to logdir

    Returns
    -------
    data_input, arch, objective, solver
    """
    model_dir = os.path.join(logdir, "model_files")
    f = os.path.join(model_dir, "data_input.py")
    # TODO: create warning if file f does not exists
    data_input = imp.load_source("input", f)
    f = os.path.join(model_dir, "architecture.py")
    arch = imp.load_source("arch", f)
    f = os.path.join(model_dir, "objective.py")
    objective = imp.load_source("objective", f)
    f = os.path.join(model_dir, "solver.py")
    solver = imp.load_source("solver", f)

    return data_input, arch, objective, solver


def load_hypes_from_logdir(logdir):
    """Load hypes from the logdir.

    Namely the modules loaded are:
    input_file, architecture_file, objective_file, optimizer_file

    Parameters
    ----------
    logdir : string
        Path to logdir

    Returns
    -------
    hypes
    """
    hypes_fname = os.path.join(logdir, "model_files/hypes.json")
    with open(hypes_fname, 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)
    _add_paths_to_sys(hypes)
    hypes['dirs']['base_path'] = logdir
    hypes['dirs']['output_dir'] = logdir

    return hypes


def create_filewrite_handler(logging_file, mode='w'):
    """
    Creates a filewriter handler.

    A copy of the output will be written to logging_file.

    Parameters
    ----------
    logging_file : string
        File to log output

    Returns
    ----------
    The filewriter handler
    """
    target_dir = os.path.dirname(logging_file)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    filewriter = logging.FileHandler(logging_file, mode=mode)
    formatter = logging.Formatter(
        '%(asctime)s %(name)-3s %(levelname)-3s %(message)s')
    filewriter.setLevel(logging.INFO)
    filewriter.setFormatter(formatter)
    logging.getLogger('').addHandler(filewriter)
    return filewriter


# Add basic configuration
def cfg():
    """General configuration values."""
    return None


def _set_cfg_value(cfg_name, env_name, default, cfg):
    """Set a value for the configuration.

    Parameters
    ----------
    cfg_name : str
    env_name : str
    default : str
    cfg : function
    """
    if env_name in os.environ:
        setattr(cfg, cfg_name, os.environ[env_name])
    else:
        logging.info("No environment variable '%s' found. Set to '%s'.",
                     env_name,
                     default)
        setattr(cfg, cfg_name, default)


_set_cfg_value('plugin_dir',
               'TV_PLUGIN_DIR',
               os.path.expanduser("~/tv-plugins"),
               cfg)
_set_cfg_value('step_show', 'TV_STEP_SHOW', 50, cfg)
_set_cfg_value('step_eval', 'TV_STEP_EVAL', 250, cfg)
_set_cfg_value('step_write', 'TV_STEP_WRITE', 1000, cfg)
_set_cfg_value('max_to_keep', 'TV_MAX_KEEP', 10, cfg)
_set_cfg_value('step_str',
               'TV_STEP_STR',
               ('Step {step}/{total_steps}: loss = {loss_value:.2f} '
                '( {sec_per_batch:.3f} sec (per Batch); '
                '{examples_per_sec:.1f} examples/sec)'),
               cfg)


def load_plugins():
    """Load all TensorVision plugins."""
    if os.path.isdir(cfg.plugin_dir):
        onlyfiles = [f for f in os.listdir(cfg.plugin_dir)
                     if os.path.isfile(os.path.join(cfg.plugin_dir, f))]
        pyfiles = [f for f in onlyfiles if f.endswith('.py')]
        import imp
        for pyfile in pyfiles:
            logging.info('Loaded plugin "%s".', pyfile)
            imp.load_source(os.path.splitext(os.path.basename(pyfile))[0],
                            pyfile)


def overlay_segmentation(input_image, segmentation, color_dict):
    """
    Overlay input_image with a hard segmentation result.

    Store the result with the same name as segmentation_image, but with
    `-overlay`.

    Parameters
    ----------
    input_image : numpy.array
        An image of shape [width, height, 3].
    segmentation : numpy.array
        Segmentation of shape [width, height].
    color_changes : dict
        The key is the class and the value is the color which will be used in
        the overlay. Each color has to be a tuple (r, g, b, a) with
        r, g, b, a in {0, 1, ..., 255}.
        It is recommended to choose a = 0 for (invisible) background and
        a = 127 for all other classes.

    Returns
    -------
    numpy.array
        The image overlayed with the segmenation
    """
    width, height = segmentation.shape
    output = scipy.misc.toimage(segmentation)
    output = output.convert('RGBA')
    for x in range(0, width):
        for y in range(0, height):
            if segmentation[x, y] in color_dict:
                output.putpixel((y, x), color_dict[segmentation[x, y]])
            elif 'default' in color_dict:
                output.putpixel((y, x), color_dict['default'])

    background = scipy.misc.toimage(input_image)
    background.paste(output, box=None, mask=output)

    return np.array(background)


def soft_overlay_segmentation(input_image,
                              seg_probability,
                              colormap=None,
                              alpha=0.4):
    """
    Overlay image with propability map.

    Overlays the image with a colormap ranging
    from blue to red according to the probability map
    given in gt_prob. This is good to analyse the segmentation
    result of a single class.

    Parameters
    ----------
    input_image : numpy.array
        Image of shape [width, height, 3]
    seg_probability : numpy.array
        Propability map for one class with shape [width, height]
    colormap : matplotlib colormap object
        Defines which floats get which color
    alpha : float
        How strong is the overlay compared to the input image


    Returns
    -------
    numpy.array
        Soft overlay of the input image with a propability map of shape
        [width, height, 3]

    Notes
    -----
    See `Matplotlib reference
    <http://matplotlib.org/examples/color/colormaps_reference.html>`_
    for more colormaps.
    """
    assert alpha >= 0.0
    assert alpha <= 1.0
    if colormap is None:
        colormap = cm.get_cmap('bwr')

    overimage = colormap(seg_probability, bytes=True)
    output = alpha * overimage[:, :, 0:3] + (1.0 - alpha) * input_image

    return output
