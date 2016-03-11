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
import logging
import scipy as scp
import scipy.misc
from six.moves import urllib

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.training import queue_runner

import params

# Global constents descriping data set

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

def input_pipeline(filename, batch_size, num_labels,
                   processing_image=lambda x:x,
                   processing_label=lambda y:y,
                   num_epochs=None):
                       
    """The input pipeline for reading images classification data.
     
    The data should be stored in a single text file of using the format:
     
     /path/to/image_0 label_0
     /path/to/image_1 label_1
     /path/to/image_2 label_2
     ...
    
     Args:
       filename: the path to the txt file
       batch_size: size of batches produced
       num_epochs: optionally limited the amount of epochs
      
    Returns:
       List with all filenames in file image_list_file
    """
    
    # Reads pfathes of images together with there labels
    image_list, label_list = read_labeled_image_list(filename)

                                                     
    images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)
    
    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=num_epochs,
                                                shuffle=True)

    # Reads the actual images from                                                 
    image, label = read_images_from_disk(input_queue,num_labels=num_labels)
    pr_image = processing_image(image)
    pr_label = processing_label(label)                             

    image_batch, label_batch = tf.train.batch([pr_image, pr_label],
                                              batch_size=batch_size)
    
    # Display the training images in the visualizer.
    tensor_name = image.op.name
    tf.image_summary(tensor_name + 'images', image_batch)                                                  
    return image_batch, label_batch


def inputs(eval_data, data_dir, batch_size, num_labels=2,num_epochs=None):

  if(eval_data):
    filename=os.path.join(data_dir, "test.txt")
  else:
    filename=os.path.join(data_dir, "test.txt")

  def pr_image(image):
    return tf.image.per_image_whitening(image)

  return input_pipeline(filename, batch_size,num_labels, processing_image=pr_image
                        ,num_epochs=None)

def distorted_inputs(data_dir, batch_size, num_labels=2, num_epochs=None):

  filename=os.path.join(data_dir, "train.txt")

  def pr_image(image):
    distorted_image = image
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

    return tf.image.per_image_whitening(distorted_image)

  return input_pipeline(filename, batch_size,num_labels, processing_image=pr_image
                        ,num_epochs=None)

        

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
  processed_example = preprocessing(example)
  # processed_labels = create_one_hot(label,num_labels)
  processed_label = label
  return processed_example, processed_label
  
  
def preprocessing(image):
    resized_image = tf.image.resize_images(image, params.image_size,
                                           params.image_size, method=0)
    resized_image.set_shape([params.image_size,params.image_size,3])
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
       label: optionally, if set label will be pasted after each line
      
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(int(label))
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
    
def test_one_hot():
    data_folder = "/fzi/ids/teichman/no_backup/DATA/"
    data_file = "Vehicle_Data/train.txt"
    
    filename = os.path.join(data_folder, data_file)
    
        # Reads pfathes of images together with there labels
    image_list, label_list = read_labeled_image_list(filename)

                                                     
    images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)
    
        # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=None,
                                                shuffle=True)

    # Reads the actual images from                                                 
    image, label = read_images_from_disk(input_queue, NUM_CLASSES)

    label_one_hot = create_one_hot(label,2)  
    
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    with sess.as_default():
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        print(sess.run([label,label_one_hot]))
        print(sess.run([label,label_one_hot]))
        print(sess.run([label,label_one_hot]))
        print(sess.run([label,label_one_hot]))
        print(sess.run([label,label_one_hot]))
        print(sess.run([label,label_one_hot]))
        

        sess.close()
    

    
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
  """Download and extract the tarball from Alex's website."""
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = params.data_url.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(params.data_url, filepath,
                                             reporthook=_progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    zipfile.ZipFile(filepath, 'r').extractall(dest_directory)
    process_data(dest_directory)

def process_data(dest_directory):
  # this are the dictionaries of data    
  path_data = os.path.join(dest_directory,"data_road/training/image_2/")
  path_gt = os.path.join(dest_directory,"data_road/training/gt_image_2/")

  logging.info("Retrieving classification samples. This will take a while.")

  # Load lists of files and names, random shuffel them
  names = [f for f in sorted(os.listdir(path_data)) if f.endswith('.png')]
  random.shuffle(names)
  num_names = len(names)

  road_snippets = os.path.join(dest_directory, "road")
  os.makedirs(road_snippets)
  bg_snippets = os.path.join(dest_directory, "background")
  os.makedirs(bg_snippets)

  train_file_name = os.path.join(dest_directory, "train.txt")
  test_file_name = os.path.join(dest_directory, "test.txt")

  train_file = open(train_file_name, "w")
  test_file = open(test_file_name, "w")

  for i, image_file in enumerate(names):

    # Copy Names of Images
    logging.info("Processing Image %i / %i : %s", i,num_names, image_file)
    data_file = os.path.join(path_data, image_file)
    gt_name = image_file.split('_')[0] + "_road_" + image_file.split('_')[1]
    gt_file = os.path.join(path_gt, gt_name)

    data = scp.misc.imread(data_file)
    gt = scp.misc.imread(gt_file)

    # mygt == 0 iff pixel is background
    mygt = 255!=np.sum(gt, axis=2)

    skip = 0      
    for x in range(mygt.shape[0]-params.image_size, 0 , -params.stride):
      for y in range(0, mygt.shape[1]-params.image_size, params.stride):
          if (skip>0):
            skip = skip-1
            continue
          if(np.sum(mygt[x:(x+params.image_size),y:(y+params.image_size)])==0):
              file_name = "%s_%i_%i.png" % (image_file.split('.')[0], x, y)
              save_file = os.path.join(bg_snippets, file_name)
              scp.misc.imsave(save_file, data[x:(x+params.image_size),y:(y+params.image_size)])
              skip = skip + 2
              if i < 200:
                train_file.write(save_file + " 0" + "\n")
              else:
                test_file.write(save_file + " 0" + "\n")
          elif(np.sum(mygt[x:(x+params.image_size),y:(y+params.image_size)]) >
               0.8*params.num_pixels):
              file_name = "%s_%i_%i.png" % (image_file.split('.')[0], x, y)
              save_file = os.path.join(road_snippets, file_name)
              scp.misc.imsave(save_file, data[x:(x+params.image_size),y:(y+params.image_size)])
              if i < 200:
                train_file.write(save_file + " 1" + "\n")
              else:
                test_file.write(save_file + " 1" + "\n")
 
    
if __name__ == '__main__':
  #test_one_hot()
  test_pipeline()
  test_pipeline()