# Braid

Braid is a flexible and modular neural network library. It is written in python and uses Tensorflow backend. It was designed for fast development and at the same time to be able to support arbitrary network designs.

The main ideas guiding the design of braid were:

**Flexibility**: There should be multiple ways to interact with the library. Design the network in Tensorflow, braid or simply provide it in a protobuf file.

**Openness**: Keep the Tensorflow backend open rather than behind layers of abstraction. It helps in modifying existing structures instead of building from ground up for unavailable network attributes and nodes.

**Modularity**: The structure of the library should not be rigid. User should be able to modify parts of library according to preference.

Braid has enough functionality built-in to support most complex network and support for building completely custom nodes and layers.

It is compatible with: Python 2.7-3.1

**Braid is useful when**:

- You require fast experimentation and don’t want to bother with boilerplate parts of the code.

- You want to customize parts of network and still want to retain the simplicity of code.
- You need to customize/streamline the entire existing library according to given specification or personal preferences.

**Installation**

*pip install braid*

If the above does not work then(requires git):

*pip install git+git://github.com/Arya-ai/braid*
