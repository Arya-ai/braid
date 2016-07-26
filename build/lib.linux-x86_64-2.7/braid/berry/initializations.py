"""Methods for weight initialization.

In order to train a deep network, we initialize the weights and biases in all
the layers with random small values. It is very important for the values to
lie in  a certain range since this mainly commands whether the model will
train effectively or not. There are mainly two problems which arise with poor
parameter initialization:

* Too large values: This leads to exploding gradients and causes a ``NaN``
  loss. After this the gradients drop to ``0``, hence rendering the training
  process useless.
* Too small values: This leads to vanishing gradients - the gradient signals
  generated are too small to make any significant change to the paramter
  values and hence, the model does not learn anything.

Below is a list of initializations defined in :mod:`berry` to help with proper
initializations based on the model architecture.

.. autosummary::
    :nosignatures:

    xavier
    deepnet


Helper function
---------------

.. autofunction:: get_initialzation

Examples
--------

You can either use the :func:`get_initialzation` function or use
the initialization function directly,

>>> from berry.initializations import xavier
>>> params = {'shape': [1500, 500], 'fan_out': 500, 'fan_in': 1500}
>>> stddev = xavier(**params)

Initializations
---------------

.. autofunction:: xavier
.. autofunction:: deepnet
"""
import tensorflow as tf
from math import sqrt

__all__ = [
    "get_initialzation"
]


def xavier(shape=None, fan_in=1, fan_out=1, **kwargs):
    """Xavier weight initialization.

    This is also known as Glorot initialization [1]_. Known to give good
    performance with sigmoid units.

    Parameters
    ----------
    shape : tuple or list
        Shape of the weight tensor to sample.

    fan_in : int
        The number of units connected at the input of the current layer.

    fan_out : int
        The number of units connected at the output of the current layer.

    Returns
    -------
    float
        Standard deviation of the normal distribution for weight
        initialization.

    References
    ----------
    .. [1] Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.

    Notes
    -----
    The weights are initialized as

    .. math::
       \\sigma &= \\sqrt{\\frac{2}{fan_{in}+fan_{out}}}\\\\
       W &\sim N(0, \\sigma)
    """
    return sqrt(2. / (fan_in + fan_out))


def deepnet(shape=None, fan_out=1, **kwargs):
    """ This initialization gives good performance with deep nets,
    e.g.: VGG-16.

    This method [1]_ was mainly developed with Relu/PRelu activations in mind
    and is known to give good performance with very deep networks which use
    these activations.

    Parameters
    ----------
    shape : tuple or list
        Shape of the weight tensor to sample.

    fan_out : int
        The number of units connected at the output of the current layer.

    Returns
    -------
    float
        Standard deviation of the normal distribution for weight
        initialization.

    References
    ----------
    .. [1] He, K., Zhang, X., Ren, S., and Sun, J. Delving Deep
           into Rectifiers: Surpassing Human-Level Performance
           on ImageNet Classification. ArXiv e-prints, February
           2015.

    Notes
    -----
    The weights are initialized as

    .. math::
        \\sigma = \\sqrt{\\frac{2}{fan_{out}}}

    """
    return sqrt(2. / (fan_out))

_INITS = {
    "xavier": xavier,
    "deepnet": deepnet
}


def get_initialzation(key):
    """Helper function to retrieve the appropriate initialization function.

    Parameters
    ----------
    key : string
        Name of the type of initialization - "xavier", "deepnet", etc.

    Returns
    -------
    function
        The appropriate function given the ``key``.

    Examples
    --------
    >>> from berry.initializations import get_initialzation
    >>> func = get_initialzation("deepnet")
    >>> params = {'shape': [1500, 500], 'fan_out': 500}
    >>> stddev = func(**params)
    """
    global _INITS
    if not _INITS.has_key(key):
        raise NotImplementedError(
            "Supported list of initialzations {}".format(_INITS.keys()))
    return _INITS[key]
