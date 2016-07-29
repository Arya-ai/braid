.. braid documentation master file, created by
   sphinx-quickstart on Fri Jul 29 14:24:38 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Braid's documentation!
=================================

Braid is a flexible and modular neural network library. It is written in python and uses Tensorflow backend. It was designed for fast development and at the same time to be able to support arbitrary network designs.

Contents:


Braid is a flexible and modular neural network library. It is written in python and uses Tensorflow backend. It was designed for fast development and at the same time to be able to support arbitrary network designs.

The main ideas guiding the design of braid were:

**Flexibility**: There should be multiple ways to interact with the library. Design the network in Tensorflow, braid or simply provide it in a protobuf file.

**Openness**: Keep the Tensorflow backend open rather than behind layers of abstraction. It helps in modifying existing structures instead of building from ground up for unavailable network attributes and nodes.

**Modularity**: The structure of the library should not be rigid. User should be able to modify parts of library according to preference.

Braid has enough functionality built-in to support most complex network and support for building completely custom nodes and layers.

It is compatible with: Python 2.7-3.0

**Braid is useful when**:

- You require fast experimentation and don’t want to bother with boilerplate parts of the code.

- You want to customize parts of network and still want to retain the simplicity of code.
- You need to customize/streamline the entire existing library according to given specification or personal preferences.

.. toctree::
   :maxdepth: 3

   user/install
   user/tutorial
   user/contributing


API Reference
-------------
If you are looking for information on a specific function, class or
method, this part of the documentation is for you.

.. toctree::
    :maxdepth: 1

    modules/berry/model
    modules/berry/layers
    modules/berry/initializations
    modules/berry/activations
    modules/berry/objectives
    modules/berry/optimizers

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

