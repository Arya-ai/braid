"""
Base layer class and helper functions to print network summary and aggregate
auxiliary inputs from a list of layers.
"""
import numpy as np
import tensorflow as tf
from math import sqrt
from ..initializations import get_initialzation
from ..utils import print_msg
from ..config import BerryKeys

__all__ = [
    "Layer",
    "get_all_aux_params",
    "print_layers_summary",
]


class Layer(object):
    """Base class for implementing a layer in :mod:`berry`.

    :class:`Layer` is a simple helper class for implementing new layers. It
    defines a set of functions which enable weight initialization, printing
    layer summary, etc. Due to ``berry``'s transparent coupling with
    tensorflow, incoming layers can be :class:`Layer` objects, or
    ``tf.Tensor`` objects.

    Parameters
    ----------
    incoming : :class:`Layer` or ``tf.Tensor``
        Parent layer, whose output is given as input to the current layer.

    W : ``tf.Variable``, optional (default = None)
        Weight tensor in case if the layer has any trainable parameters.

    b : ``tf.Variable``, optional (default = None)
        Bias vector in case of trainable parameters.

    name : string, optional (default = None)
        Name of the layer. Should be specified for better readability.

    Attributes
    ----------
    input_layer : :class:`Layer` or ``tf.Tensor``
        Input layer to this layer.

    input_shape : tuple
        Shape of the incoming layer.

    output : ``tf.Tensor``
        The Tensor obtained after performing the transformation applied by
        this layer.

    output_shape : tuple
        Shape of the output tensor.

    type : string
        Return the name of the class.
    """

    def __init__(self, incoming, W=None, b=None, name=None):
        if isinstance(incoming, tf.Variable) or isinstance(
                incoming, tf.Tensor):
            self.input_shape = tuple(incoming.get_shape().as_list())
            self.input_layer = incoming
            self.layer_index = 1
        else:
            self.input_shape = incoming.output_shape
            self.input_layer = incoming.output
            self.layer_index = incoming.layer_index + 1
        self._output = None
        if W is not None and not isinstance(W, tf.Variable):
            print_msg("W should be a ``tensorflow.Tensor``", obj=self)
        if b is not None and not isinstance(b, tf.Variable):
            print_msg("b should be a ``tensorflow.Tensor``", obj=self)
        self.W = W
        self.b = b
        if not self.validate_input_layer(incoming):
            print_msg("Invalid input layer", obj=self)

        self.params = []
        self.name = name if name else 'layer_{}'.format(self.layer_index)

    @property
    def output_shape(self):
        return self.get_output_shape_for(self.input_shape)

    @property
    def output(self):
        if self._output is None:
            self._output = self.get_output_for()
        return self._output

    @property
    def type(self):
        return self.__class__.__name__

    def weight_variable(self, shape, initializer=None, stddev=0.01):
        """Create a weight tensor with appropriate initialization.

        Parameters
        ----------
        shape : list
            Shape of the weight tensor to create.

        initializer : string, optional (default = None)
            Initialize using a pre-defined method in
            :mod:`berry.initializations`.

        stddev : float, optional (default = 0.01)
            Standard deviation of Gaussian distribution for initialization.

        Returns
        -------
        ``tf.Variable``
            Symbolic weight tensor initialized according to ``initializer``
            method specified or from a Gaussian distribution with a mean
            of ``0`` and a standard deviation of ``stddev``.
        """
        if self.W is not None:
            if self.W.get_shape().as_list() != list(shape):
                print_msg("shape mismatch of `W` provided and expected",
                          obj=self)
            return self.W
        if initializer is None:
            std_dev = stddev
        else:
            func_init = get_initialzation(initializer)
            param = {
                'shape': self.get_W_shape(),
                'fan_in': self.get_fan_in(),
                'fan_out': self.get_fan_out()
            }
            std_dev = func_init(**param)
        print_msg("std dev: {}".format(std_dev), obj=self, level='info')
        initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=std_dev)
        return tf.Variable(initial, name='weights')

    def bias_variable(self, shape, initializer=None, val=0.0):
        """Create a bias vector with appropriate initialization.

        Parameters
        ----------
        shape : list
            Shape of the weight tensor to create.

        initializer : string, optional (default = None)
            Not used

        val : float, optional (default = 0.0)
            Constant value for initializing the biases.

        Returns
        -------
        ``tf.Variable``
            Symbolic bias vector initialized with a value of ``val``.
        """
        if self.b is not None:
            if self.W.get_shape().as_list() != list(shape):
                print_msg("shape mismatch of `b` provided and expected",
                          obj=self)
            return self.b
        initial = tf.constant(val, dtype=tf.float32, shape=shape)
        return tf.Variable(initial, name='biases')

    def get_W_shape(self):
        """Shape of the weight tensor required for the layer.

        Notes
        -----
        .. note::

            Should be **overwritten** for layers which have trainable weight
            tensors.
        """
        return []

    def get_b_shape(self):
        """Number of bias units

        Notes
        -----
        .. note::

            Should be **overwritten** for layers which have trainable weight
            tensors.
        """
        return []

    def get_fan_in(self):
        """Number of input units to the layer.

        Returns
        -------
        int
            The fan in of the layer.

        Notes
        -----
        .. note::

            Should be **overwritten** if the layer has trainable weights.
        """
        return NotImplementedError

    def get_fan_out(self):
        """Number of output units to the layer.

        Returns
        -------
        int
            The fan out of the layer.

        Notes
        -----
        .. note::

            Should be **overwritten** if the layer has trainable weights.
        """
        return NotImplementedError

    def get_output_shape_for(self, input_shape):
        """Shape of the output tensor produced by this layer.

        This method should be **overwritten** in the inherited layer.

        Parameters
        ----------
        input_shape : tuple or list
            Shape of the input layer.

        Returns
        -------
        tuple
            Shape of the output tensor.

        Notes
        -----
        .. note::

            By default, this will return the ``input_shape``.
        """
        return input_shape

    def get_output_for(self):
        """Compute the output transformation on the input tensor.

        The main logic of the layer lies here. The required sequence of
        tensorflow operations for transforming the input to output are
        defined here. This should be **overwritten** in the inherited layer.

        Returns
        -------
        ``tf.Tensor``
            Output tensor of this layer.

        Notes
        -----
        .. note::

            For easy access to all the layer pre-activations/outputs, the Tensor objects are added to a `tensorflow collection <https://www.tensorflow.org/versions/r0.9/api_docs/python/framework.html#graph-collections>`_ with ``BerryKeys.LAYER_OUTPUTS`` key. Example:

            >>> # Add to collection
            >>> tf.add_to_collection(BerryKeys.LAYER_OUTPUTS, output)
            >>> # Retrieve all from the collection
            >>> vars = tf.get_collection(BerryKeys.LAYER_OUTPUTS)
        """
        raise NotImplementedError

    def validate_input_layer(self):
        """This function ensures valid layer-layer connections are made.

        Typically, this function would perform assertions on the shape of the
        input layer. This function must be **overwritten** in the inherited
        class.

        Returns
        -------
        bool
            ``True`` if input layer is valid, ``False`` otherwise.
        """
        return True

    def print_summary(self, print_fmt):
        """Print out a summary of the layer - layer type, name, weight shape,
        input shape and output shape.

        Parameters
        ----------
        print_fmt : string
            Formatted string with 5 fields.

        Examples
        --------
        >>> fmt = "|{:^20}|{:^15}|{:^20}|{:^20}|{:^20}|"
        >>> print l.print_summary(fmt)
        |   Convolution2D    |     conv1     |   [5, 5, 3, 64]    |(None, \\
        224, 224, 3) |(None, 113, 113, 64)|
        """
        weight_shape = self.get_W_shape()
        weight_shape = None if len(weight_shape) == 0 else weight_shape
        print print_fmt.format(self.type, self.name, weight_shape,
                               self.input_shape, self.output_shape)


def get_all_aux_params(deterministic):
    """Aggregate all auxiliary parameters from all the layers for training and
    testing.

    Parameters
    ----------
    deterministic : bool
        ``True`` for test phase and ``False`` for train phase.

    Returns
    -------
    dict
        Dictionary with parameter name as "key" and their value as "value".
    """
    feeder = {}
    cols = tf.get_collection(BerryKeys.AUX_INPUTS)
    for t in cols:
        if hasattr(t, 'train_val') and hasattr(t, 'test_val'):
            feeder[t.name] = t.train_val if not deterministic else t.test_val
    return feeder


def print_layers_summary(layers_list):
    """Print a nice formatted summary of the network.

    Parameters
    ----------
    layers_list : list of :class:`Layer`
        List of layers.
    """
    print_msg("Network summary", level='info')
    print_fmt = "|{:^20}|{:^15}|{:^20}|{:^20}|{:^20}|"
    n = len(print_fmt.format("1", "1", "1", "1", "1"))
    line = ''.join(["-" for i in range(n)])
    print line
    print print_fmt.format(
        "Layer Name", "Name", "Weight shape", "Input shape", "Output shape")
    print line
    for lay in layers_list:
        if isinstance(lay, (tf.Tensor, tf.Variable)):
            continue
        lay.print_summary(print_fmt)
        print line
