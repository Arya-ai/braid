"""Nonlinear activations applied after each layer.

Nonlinearities are responsible for the effectiveness of neural networks. They allow us to string together a long chain of layers and learn interesting, non-linear and highly complex functions.

The list of supported nonlinearities are:

- relu
- softplus
- sigmoid
- tanh
- softmax
- linear

Helper function
---------------

.. autofunction:: get_activation
"""
import tensorflow as tf


__all__ = [
    "get_activation"
]


_ACTIVATIONS = {
    "relu": tf.nn.relu,
    "softplus": tf.nn.softplus,
    "sigmoid": tf.sigmoid,
    "tanh": tf.tanh,
    "softmax": tf.nn.softmax,
    "linear": None
}


def get_activation(key):
    """Helper function to retrieve the appropriate activation function.

    Parameters
    ----------
    key : string
        Name of the type of activation - "relu", "sigmoid", etc.

    Returns
    -------
    function
        The appropriate function given the ``key``.
    """
    global _ACTIVATIONS
    if not _ACTIVATIONS.has_key(key):
        raise NotImplementedError(
            "Supported list of activations {}".format(_ACTIVATIONS.keys()))
    return _ACTIVATIONS[key]
