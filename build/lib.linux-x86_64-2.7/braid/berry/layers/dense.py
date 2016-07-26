"""Dense or fully connected layer
"""
from __future__ import division
import tensorflow as tf
import math
import warnings
from .base import Layer
from ..activations import get_activation
from .. import initializations as init
from ..utils import print_activations
from ..config import BerryKeys

__all__ = [
    "Dense"
]


class Dense(Layer):
    """Fully connected or dense layer

    Parameters
    ----------
    incoming : :class:`Layer` or ``tf.Tensor``
        Parent layer, whose output is given as input to the current layer.

    num_units : int
        The number of hidden units.

    activation : string, optional (default = "linear")
        Nonlinearity to apply afer performing convolution. See
        :mod:`berry.activations`

    init : string, optional (default = None)
        Weight initialization method to choose. See
        :mod:`berry.initializations`

    W_stddev : float, optional (default = 1e-2)
        Standard deviation for Normal distribution to initialize the weights,
        if ``init = None``.

    b_val : float, optional (default = 0.1)
        Constant value to initialize the biases.

    W : ``tf.Variable``, optional (default = None)
        Weight tensor (inherited from :class:`Layer`).

    b : ``tf.Variable``, optional (default = None)
        Bias vector (inherited from :class:`Layer`).

    name : string, optional (default = None)
        Name of the layer. Should be specified for better readability (
        inherited from :class:`Layer`).

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

    Notes
    -----
    .. seealso:: Inherits class :class:`Layer`.
    """

    def __init__(self, incoming, num_units, activation='linear',
                 init=None, W_stddev=1e-3, b_val=0.1, **kwargs):
        super(Dense, self).__init__(incoming, **kwargs)
        self.activation = get_activation(activation)
        self.num_units = int(num_units)
        if init is None and W_stddev == 1e-3 and b_val == 0.1:
            warnings.warn(
                "[WARN] {}: using default `W_stddev` ({})"
                " and `b_val` ({}) values for weight init.".format(
                    self.type, W_stddev, b_val))
        self.init = init
        self.W_stddev = W_stddev
        self.b_val = b_val

    def get_W_shape(self):
        """Shape of the weight tensor

        Returns
        -------
        list
            ``[input_channels, num_units]``
        """
        return [self.input_shape[-1], self.num_units]

    def get_b_shape(self):
        """Number of bias units

        Returns
        -------
        list
            ``[num_units]``
        """
        return [self.num_units]

    def get_fan_in(self):
        """Input receptive field

        Returns
        -------
        int
            ``input_channels``
        """
        return self.input_shape[-1]

    def get_fan_out(self):
        """Output receptive field

        Returns
        -------
        int
            ``num_units``
        """
        return self.num_units

    def get_output_shape_for(self, input_shape):
        """Shape of the output tensor

        Parameters
        ----------
        input_shape : tuple or list
            Shape of the input layer.

        Returns
        -------
        tuple
            Shape of the output tensor. ``(batch_size, num_units)``
        """
        return (input_shape[0], self.num_units)

    def validate_input_layer(self, incoming):
        """Validate the input layer shape

        Returns ``True`` if the input layer is 2D else, raise an
        :py:exc:`exceptions.AssertError`.
        """
        assert len(self.input_shape) == 2, (
            "[{}] Input shape error: Dense layer "
            "requires input shape: (batch_size, num_units)".format(self.type))
        return True

    def get_output_for(self):
        """Perform the matrix product, add the bias, apply the activation
        function and return the output ``tf.Tensor``

        Returns
        -------
        ``tf.Tensor``
            Output tensor of this layer.
        """
        with tf.name_scope(self.name) as scope:
            # define layer weights and biases
            weights = self.weight_variable(
                self.get_W_shape(),
                initializer=self.init,
                stddev=self.W_stddev)
            biases = self.bias_variable(
                self.get_b_shape(),
                initializer=self.init,
                val=self.b_val)

            if self.activation is None:
                preactivation = tf.nn.bias_add(
                    tf.matmul(self.input_layer, weights), biases, name=scope)
                output = preactivation
            else:
                preactivation = tf.nn.bias_add(
                    tf.matmul(self.input_layer, weights), biases)
                output = self.activation(preactivation, name=scope)
            self.params = [weights, biases]
        tf.add_to_collection(BerryKeys.LAYER_OUTPUTS, output)
        return output
