"""
Pooling layers
"""
from __future__ import division
import tensorflow as tf
import warnings
from .base import Layer
from ..activations import get_activation
from .. import initializations as init
from ..utils import print_activations, get_convolve_shape
from ..config import BerryKeys

__all__ = [
    "MaxPooling2D"
]


class MaxPooling2D(Layer):
    """Pooling using the ``max`` operation

    This is used to subsample the output from convolutional layers. It works
    by convolving a kernel with a ``max`` operation across the activation of
    the previous layer. More specifically, it looks at a receptive field
    equal to ``[1, kernel_size, kernel_size, 1]`` and outputs the ``max``
    value of the activations in this region, shifting then by ``stride``
    amount and repeating.

    Parameters
    ----------
    incoming : :class:`Layer` or ``tf.Tensor``
        Parent layer, whose output is given as input to the current layer.

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

    def __init__(self, incoming, kernel_size,
                 stride=1, pad='VALID', **kwargs):
        super(MaxPooling2D, self).__init__(incoming, **kwargs)
        if not (isinstance(kernel_size, int) and isinstance(stride, int)):
            raise NotImplementedError("[{}] only symmetric kernels and "
                                      "strides supported; kernel_size should "
                                      "be `int`".format(self.type))
        self.kernel_size = [1, kernel_size, kernel_size, 1]
        self.stride = [1, stride, stride, 1]
        pad = pad.upper()
        if pad not in {'VALID', 'SAME'}:
            raise ValueError("[{}] `pad` expects either 'VALID'"
                             " or 'SAME' as value.".format(self.type))
        if pad == 'SAME' and self.kernel_size[0] % 2 == 0:
            raise NotImplementedError("[{}] 'SAME' padding requires "
                                      "odd kernel size.".format(self.type))
        self.pad = pad

    def get_output_shape_for(self, input_shape):
        """Shape of the output tensor after performing MaxPooling.

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
                for i, k, s in zip(input_shape[1:],
                                   self.kernel_size[1:],
                                   self.stride[1:])
            ]
        )
        assert len(shape) == 4, (
            "[{}] Output shape error: should be 4D.".format(self.type))
        return shape

    def validate_input_layer(self, incoming):
        """Validate the input layer dimensions.

        Valid input layer would be 4D.

        Parameters
        ----------
        incoming : :class:`Layer` or ``tf.Tensor``
            Parent layer, whose output is given as input to the current layer.

        Returns
        -------
        bool
            ``True`` if connection is valid or raises an
            :py:exc:`exceptions.AssertionError`.
        """
        assert len(self.input_shape) == 4, (
            "[{}] Input shape error: 4D input reqd.".format(self.type))
        return True

    def get_output_for(self):
        """Perform the max pooling operation and return the output
        ``tf.Tensor``.

        Returns
        -------
        ``tf.Tensor``
            Output tensor of this layer.
        """
        with tf.name_scope(self.name) as scope:
            output = tf.nn.max_pool(
                self.input_layer,
                ksize=self.kernel_size,
                strides=self.stride,
                padding=self.pad,
                name=scope
            )
        tf.add_to_collection(BerryKeys.LAYER_OUTPUTS, output)
        return output
