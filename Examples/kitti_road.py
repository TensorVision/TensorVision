# -*- coding: utf-8 -*-
"""This file contains all model parameters"""

# Specify Model to be trained
input_file = "Examples/Inputs/kitti_road_input.py"
network_file = "Examples/Networks/minimal_cnn.py"
opt_file = "Examples/Optimizer/adam.py"


# Optionally: Where to download data
data_url = "http://kitti.is.tue.mpg.de/kitti/data_road.zip"

# Data Params
stride = 10
num_classes = 2
image_size = 50
num_channels = 3
num_pixels = image_size*image_size

# Global constants describing the Kitti data set.
num_examples_per_epoch_for_train = 100000
num_examples_per_epoch_for_eval = 65400


# Train Params
batch_size = 128

max_steps = 200000   # Number of steps to run trainer.
learning_rate = 0.01 # Learning rate 