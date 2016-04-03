# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:50:47 2015

@author: teichman
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import numpy as np


import gzip
import os
import re
import sys
import zipfile
import random
import math
import logging
import scipy as scp
import scipy.misc
from six.moves import urllib

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.training import queue_runner

from tensorflow.python.ops import random_ops

import params

# Global constents descriping data set

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

def input_pipeline(filename, batch_size, num_classes=num_classes
                   processing_image=lambda (x,y) : (x,y),
                   num_epochs=None):
                       
    """The input pipeline for reading images segmentation data.
     
    The data should be stored in a single text file of using the format:
     
     /path/to/image_0 /path/to/label_0
     /path/to/image_1 /path/to/label_1
     /path/to/image_2 /path/to/label_2
     ...

     Image and Labels will have the same dimenstion.
    
     Args:
       filename: the path to the txt file
       batch_size: size of batches produced
       num_epochs: optionally limited the amount of epochs
      
    Returns:
       image_batch: batch which images
       label_batch: batch which labels
    """
    
    # Reads pathes of images together with there labels
    image_list, label_list = read_labeled_image_list(filename)

                                                     
    images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    labels = ops.convert_to_tensor(label_list, dtype=dtypes.string)
    
    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=num_epochs,
                                                shuffle=True)

    # Reads the actual images from                                                 
    image, label = read_images_from_disk(input_queue, num_labels=num_labels)
    pr_image, pr_label = processing_image(image, label)

    image_batch, label_batch = tf.train.batch([pr_image, pr_label],
                                              batch_size=batch_size)
    
    # Display the training images in the visualizer.
    tensor_name = image.op.name
    tf.image_summary(tensor_name + 'images', image_batch)                                                  
    return image_batch, label_batch


def inputs(eval_data, data_dir, batch_size, num_labels=2,num_epochs=None):

  if hasattr(params, 'data_dir'):
    data_dir = params.data_dir

  if(eval_data):
    filename=os.path.join(data_dir, params.val_file)
  else:
    filename=os.path.join(data_dir, params.train_file)

  def pr_image(image):

    resized_image = tf.image.resize_images(image, params.image_size,
                                           params.image_size, method=0)    
    cropped_image = tf.image.resize_image_with_crop_or_pad(resized_image,
                                                           params.input_size,
                                                           params.input_size)

    cropped_image.set_shape([params.input_size,
                             params.input_size,
                             params.num_channels])

    return tf.image.per_image_whitening(cropped_image)

  return input_pipeline(filename, batch_size,num_labels, processing_image=pr_image
                        ,num_epochs=None)

def distorted_inputs(data_dir, batch_size, num_labels=2, num_epochs=None):
   """ Function creates input for training. It uses input the input pipeline and 
   """

  if hasattr(params, 'data_dir'):
    data_dir = params.data_dir

  filename=os.path.join(data_dir, "train.txt")

  def pr_image(image):

    reshaped_image = random_resize(image, params.min_scale,
                                   params.max_scale)

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image,
                                     [params.input_size, params.input_size
                                     ,params.num_channels])

    distorted_image.set_shape([params.input_size,
                               params.input_size,
                               params.num_channels])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)
    distorted_image = tf.image.random_hue(distorted_image,
                                          max_delta=0.2)

    distorted_image = tf.image.random_saturation(distorted_image,
                                                 lower=0.5,
                                                 upper=1.5)


    return tf.image.per_image_whitening(distorted_image)

  return input_pipeline(filename, batch_size,num_labels, processing_image=pr_image
                        ,num_epochs=None)


def random_resize(image, lower_size, upper_size):
  """Randomly resizes an image 

  Args:
    lower_size:
    upper_size:

  Returns:
    a randomly resized image
  """

  new_size = tf.to_int32(random_ops.random_uniform([], lower_size, upper_size))

  return tf.image.resize_images(image, new_size, new_size,
                                method=0)


        

def read_images_from_disk(input_queue, num_labels):
  """Consumes a single filename and label as a ' '-delimited string.

  Args:
    filename_and_label_tensor: A scalar string tensor.

  Returns:
    Two tensors: the decoded image, and the string label.
  """
  label = input_queue[1]
  file_contents = tf.read_file(input_queue[0])
  example = tf.image.decode_png(file_contents, channels=3)
  example = rescale_image(example)
  # processed_label = label
  return example, label
  
  
def rescale_image(image):
    """Resizes the images.

    Args:
    image: An image tensor.

    Returns:
    An image tensor with size params.image_size
    """
    resized_image = tf.image.resize_images(image, params.image_size,
                                           params.image_size, method=0)
    resized_image.set_shape([params.image_size, params.image_size,
                             params.num_channels])
    return resized_image
    


def create_one_hot(label, num_labels = 10):
    """Produces one_hot vectors out of numerical labels
    
    Args:
       label_batch: a batch of labels
       num_labels: maximal number of labels
      
    Returns:
       Label Coded as one-hot vector
    """

    labels = tf.sparse_to_dense(label, [num_labels], 1.0, 0.0)
    
    return labels



def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
    
    Args:
       image_list_file: a .txt file with one /path/to/image per line
      
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(label)
    return filenames, labels
        


        
def create_input_queues(image, label, capacity=100):
    """Creates Queues a FIFO Queue out of Input tensor objects.
     
     This function is no longer used in the input pipeline.
     However it took me a while to understand queuing and it might be useful
     fot someone at some point.

    Args:
       image: an image tensor object, generated by queues.
       label: an label tensor object, generated by queues.
      
    Returns: Two FiFO Queues
    """
    
    #create input queues

    im_queue = tf.FIFOQueue(capacity, dtypes.uint8)
    enqueue_op = im_queue.enqueue(image)
    
    queue_runner.add_queue_runner(queue_runner.QueueRunner(im_queue,
                                                           [enqueue_op]))

    label_queue = tf.FIFOQueue(capacity, dtypes.uint8)
    enqueue_op = label_queue.enqueue(label)
    
    queue_runner.add_queue_runner(queue_runner.QueueRunner(label_queue,
                                                           [enqueue_op]))
                                                           
    return im_queue, label_queue
    

    
def test_pipeline():
    data_folder = "/fzi/ids/teichman/no_backup/DATA/"
    data_file = "Vehicle_Data/test.txt"
    
    filename = os.path.join(data_folder, data_file)
    
    image_batch, label_batch = inputs(filename, 75,2)
    
    
    
    # Create the graph, etc.
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    with sess.as_default():
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        print(label_batch.eval())

        coord.request_stop()
        coord.join(threads)

        print("Finish Test")        
    
        sess.close()

def maybe_download_and_extract(dest_directory):
  """Download and extract Data found in data_url."""
  return
 
    
if __name__ == '__main__':
  #test_one_hot()
  test_pipeline()
  test_pipeline()