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
* --config=Examples/myconfig
* --name=myname


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
TensorVision to your PYTHONPATH. In my case, that is:

```bash
$ export PYTHONPATH="${PYTHONPATH}:/home/moose/GitHub/TensorVision/"
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
    "optimizer_file" : "examples/optimizer/exp_decay.py"
  }
}
```


#### Adjust input file

The `input_file` contains the path to a Python file. This Python file has to
have a function `inputs(hypes, q, phase, data_dir)`.


#### Adjust architecture file

The `architecture_file` contains the architecture of the network. It has to
have the following functions:

* `loss(H, logits, labels)`
* `inference(H, images, train=True)`
* `evaluation(H, logits, labels)`


#### Adjust the solver file

The `optimizer_file` contains the path to a Python file. This Python file has
to have a function `training(H, loss, global_step)`. It defines how one tries
to find a minimum of the loss function.
