"""Convolutional layers
"""
from __future__ import division
import tensorflow as tf
import warnings
from .base import Layer
from ..activations import get_activation
from .. import initializations as init
from ..utils import print_activations, get_convolve_shape, print_msg
from ..config import BerryKeys

# __all__ = [
#     "Convolution2D"
]

__all__ = [
    "RNN",

]

class RNN(Layer):
    """2D convolution layer

    Parameters
    ----------
    incoming : :class:`Layer` or ``tf.Tensor``
        Parent layer, whose output is given as input to the current layer.

    num_filters : int
        The number of filters to learn.

    kernel_size : int
        The size of the kernel to consider for pooling.

    stride : int, optional (default = 1)
        The amount of subsample.

    pad : string, optional (default = "VALID")
        Type of padding to apply to the input layer before doing pooling.
        Expected values - "VALID", "SAME". No padding is applied for "VALID",
        while a padding of ``(kernel_size + 1) / 2`` if "SAME". This
        ensures that the output layer shape is the same as that of the input
        layer shape.

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
        Weight tensor in case if the layer has any trainable parameters
        (inherited from :class:`Layer`).

    b : ``tf.Variable``, optional (default = None)
        Bias vector in case of trainable parameters (inherited from
        :class:`Layer`).

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

    def __init__(self, incoming, num_units,num_of_cells,return_cell_out = False, use_peepholes=False, cell_clip=None, pad='VALID', activation='linear',
                 init=None, W_stddev=1e-2, b_val=0.1, **kwargs):
        super(RNN, self).__init__(incoming, **kwargs)
        self.activation = get_activation(activation)
        if init is None and W_stddev == 1e-2 and b_val == 0.1:
            print_msg(
                "using default ``W_stddev`` ({})"
                " and ``b_val`` ({}) values for weight init.".format(
                    W_stddev, b_val), obj=self, level='warn')
        self.init = init
        self.W_stddev = W_stddev
        self.b_val = b_val
        self.num_units = num_units
        self.num_of_cells = num_of_cells
        self.return_cell_out = return_cell_out

        # pad = pad.upper()
        # if pad not in {'VALID', 'SAME'}:
        #     raise ValueError("[{}] ``pad`` expects either 'VALID'"
        #                      " or 'SAME' as value.".format(self.type))

        # if pad == 'SAME' and (
        #         self.kernel_size[0] % 2 == 0 or self.kernel_size[1] % 2 == 0):
        #     raise NotImplementedError("[{}] 'SAME' padding requires "
        #                               "odd kernel size.".format(self.type))
        # self.pad = pad

    
    def validate_input_layer(self, incoming):
        """Validate the input layer shape

        Returns ``True`` if the input layer is 4D else, raise an
        :py:exc:`exceptions.AssertError`.
        """
        assert len(self.input_shape) == 4, (
            "{} Input shape error: 2D convolution "
            "requires input shape: (batch_size, "
            "height, width, channels)".format(self.type))
        return True

    def get_output_shape_for(self, input_shape):
        """Shape of the output tensor after performing convolution.

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

            Each dimension of the output is given as

            .. math::

                l_o = \\frac{( W - F +2P)}{S} + 1

            where :math:`W` is the width of the input layer dim, :math:`F` is the ``kernel_size``, :math:`P` is the amount of padding applied and :math:`S` is the ``stride``.
        """
        batch_size = input_shape[0]
        shape = tuple(
            [batch_size, ] + [
                get_convolve_shape(i, k, s, self.pad)
                for i, k, s in zip(input_shape[1:3],
                                   self.kernel_size,
                                   self.stride[1:3])
            ] + [self.num_filters, ]
        )
        assert len(shape) == 4, (
            "{} Output shape error: should be 4D.".format(self.type))
        return shape

    def get_output_for(self):
        """Perform the convolution operation, activation and return the output
        ``tf.Tensor``.

        Returns
        -------
        ``tf.Tensor``
            Output tensor of this layer.
        """
        with tf.name_scope(self.name) as scope:
            # define layer weights and biases
            kernel = self.weight_variable(
                self.get_W_shape(),
                initializer=self.init,
                stddev=self.W_stddev)
            biases = self.bias_variable(
                self.get_b_shape(),
                initializer=self.init,
                val=self.b_val)
            conv = tf.nn.conv2d(self.input_layer, kernel,
                                self.stride, padding=self.pad)
            if self.activation is None:
                preactivation = tf.nn.bias_add(conv, biases, name=scope)
                output = preactivation
            else:
                preactivation = tf.nn.bias_add(conv, biases)
                output = self.activation(preactivation, name=scope)
            self.params = [kernel, biases]
        tf.add_to_collection(BerryKeys.LAYER_OUTPUTS, output)
        return output
