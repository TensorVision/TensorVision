[![Documentation Status](https://readthedocs.org/projects/tensorvision/badge/?version=latest)](http://tensorvision.readthedocs.org/en/latest/?badge=latest)
[![Code Issues](https://www.quantifiedcode.com/api/v1/project/3ef49e94e03a42b0bf896f5377c8e741/badge.svg)](https://www.quantifiedcode.com/app/project/3ef49e94e03a42b0bf896f5377c8e741)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/
TensorVision/TensorVision/blob/master/LICENSE)

A simple toolkit to easily apply image classification networks.


## Usage

Train a model:

```bash
$ python train.py
```

Evaluate a model:

```bash
$ python eval.py
```

Flags:
* --hypes=myconfig.json
* --name=myname


## Configurations

TensorVision comes with reasonable defaults. You only need to read this if you
want tweak it to your needs.

TensorVision is configured with environment variables. It is quite easy to
set them yourself (see [multiple ways](http://unix.stackexchange.com/a/117470/4784)).
The supported variables are:

* `TV_DIR_DATA`: The default directory where to look for data.
* `TV_DIR_RUNS`: The default directory where to look for the model.
* `TV_IS_DEV`: Either 0 or 1 - set if you want to see debug messages.
* `TV_PLUGIN_DIR`: Directory with Python scripts which will be loaded by utils.py
* `TV_SAVE`: Whether to keep all runs on default. By default runs will be written to `TV_DIR_RUNS/debug` and overwritten by newer runs, unless `tv-train --save` is called. 
* `TV_STEP_SHOW`: After how many epochs of training shot the `TV_STEP_STR` be
                  printed?
* `TV_STEP_STR`: Set what you want to see each 100th step of the training.
  The default is `'Step {step}/{total_steps}: loss = {loss_value:.2f} ( {sec_per_batch:.3f} sec (per Batch); {examples_per_sec:.1f} examples/sec;)'`
                   imported by TensorVision
* `TV_USE_GPUS`: Controll which gpus to use. Default all GPUs are used, GPUs can be specified using `--gpus`. Setting `TV_USE_GPUS='force'` makes the flag `--gpus` compulsory, this is useful in cluster environoments. Use `TV_USE_GPUS='0,3'` to run Tensorflow an the GPUs with ids 0 and 3. 


## Requirements

The following python moduls need to be installed:

* tensorflow 0.71
* numpy
* scipy
* PIL / pillow


You can do that by

```bash
$ pip install -r requirements.txt
```


## Development

First of all, you should install the additional development requirements:

```bash
$ pip install -r requirements-dev.txt
```

For development, you can avoid reinstalling the TensorVision package by adding
TensorVision to your PYTHONPATH and the `bin` directory to your PATH. In my
case, that is:

```bash
$ export PYTHONPATH="${PYTHONPATH}:/home/moose/GitHub/TensorVision/"
$ export PATH="/home/moose/GitHub/TensorVision/bin/:$PATH"
```

You can run the tests by

```bash
$ python setup.py test
```


## Workflow

### One-time stuff

* Create a config.py for general configuration

### Each time

Each time you get a new task


#### Create JSON file

Create a json file (e.g. `cifar10_cnn.json`). It has at least the following
content:

```json
{
  "model": {
    "input_file": "examples/inputs/cifar10_input.py",
    "architecture_file" : "examples/networks/cifar_net.py",
    "objective_file" : "examples/objectives/softmax_classifier.py",
    "optimizer_file" : "examples/optimizer/exp_decay.py"
  }
}
```


#### Adjust input file

The `input_file` contains the path to a Python file. This Python file has to
have a function `inputs(hypes, q, phase, data_dir)`.


#### Adjust architecture file

The `architecture_file` contains the architecture of the network. It has to
have the function `inference(hypes, images, train=True)`, which takes image tensors
creates a computation graph to produce logits

#### Adjust objective file

The `objective_file` contains task spezific code od the model. It
has to implement the following functions:

* `decoder(hypes, images, train=True)`
* `loss(hypes, decoder, labels)`
* `evaluation(hypes, decoder, labels)`


#### Adjust the solver file

The `optimizer_file` contains the path to a Python file. This Python file has
to have a function `training(H, loss, global_step)`. It defines how one tries
to find a minimum of the loss function.
