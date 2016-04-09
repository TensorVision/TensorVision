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
