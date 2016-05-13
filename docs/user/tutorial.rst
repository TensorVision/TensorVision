.. _tutorial:

========
Tutorial
========

This tutorial introduces the general workflow when using TensorVision.
Examples can be found in the `Modell Zoo`_ repository.

Basics
======

Train a model:


.. highlight:: bash

  tv-train --hypes config.json


Evaluate a model:


.. highlight:: bash

  python eval.py


Flags:

* ``--hypes=myconfig.json``
* ``--name=myname``


Workflow
========

Each time you get a new task


Create JSON file
----------------

Create a json file (e.g. `cifar10_cnn.json`). It has at least the following
content:

.. highlight:: json

  {
    "model": {
      "input_file": "examples/inputs/cifar10_input.py",
      "architecture_file" : "examples/networks/cifar_net.py",
      "objective_file" : "examples/objectives/softmax_classifier.py",
      "optimizer_file" : "examples/optimizer/exp_decay.py"
    }
  }


Adjust input file
-----------------

The ``input_file`` contains the path to a Python file. This Python file has to
have a function ``inputs(hypes, q, phase, data_dir)``.

The parameters of `inputs` are:

* ``hypes``: A dictionary which contains everything your `model.json` file had.
* ``q``: A queue (e.g. `FIFOQueue`_
* ``phase``: Either ``train`` or ``val``
* ``data_dir``: Path to the data. This can be set with ``TV_DIR_DATA``.

The expected return value is a tuple (xs, ys), where x is a list of features
and y is a list of labels.


Adjust architecture file
------------------------

The ``architecture_file`` contains the architecture of the network. It has to
have the function ``inference(hypes, images, train=True)``, which takes image
tensors creates a computation graph to produce logits


Adjust objective file
---------------------

The ``objective_file`` contains task spezific code od the model. It
has to implement the following functions:

* ``decoder(hypes, images, train=True)``
* ``loss(hypes, decoder, labels)``
* ``evaluation(hypes, decoder, labels)``


Adjust the solver file
----------------------

The ``optimizer_file`` contains the path to a Python file. This Python file has
to have a function ``training(H, loss, global_step)``. It defines how one tries
to find a minimum of the loss function.



.. _Modell Zoo: https://github.com/TensorVision/modell_zoo
.. _FIFOQueue : https://www.tensorflow.org/versions/r0.8/how_tos/threading_and_queues/index.html