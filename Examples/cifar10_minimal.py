# -*- coding: utf-8 -*-
"""This file contains all model parameters"""

# Specify Model to be trained
input_file = "Examples/Inputs/cifar10_input.py"
network_file = "Examples/Networks/minimal_cnn.py"
opt_file = "Examples/Optimizer/adam.py"


# Optionally: Where to download data
data_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

# Data Params
crop_size = 32 
image_size = 24
num_classes = 10
num_channels = 3

num_examples_per_epoch_for_train = 50000
num_examples_per_epoch_for_eval = 10000


# Train Params
batch_size = 128

max_steps = 200000   # Number of steps to run trainer.
learning_rate = 0.01 # Learning rate 