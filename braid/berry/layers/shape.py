from __future__ import division
import tensorflow as tf
import numpy as np
import math
from .base import Layer
from .. import activations
from .. import initializations as init
from ..utils import print_activations
from ..config import BerryKeys

__all__ = [
    "Flatten"
]


class Flatten(Layer):
    """Flatten layer

    Parameters
    ----------
    incoming : :class:`Layer` or ``tf.Tensor``
        Parent layer, whose output is given as input to the current layer.

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

    def __init__(self, incoming, **kwargs):
        super(Flatten, self).__init__(incoming, **kwargs)

    def validate_input_layer(self, incoming):
        """Validate the input layer shape

        Returns ``True`` if the input layer is 2+D else, raise an
        :py:exc:`exceptions.AssertError`.
        """
        assert len(self.input_shape) > 2, (
            "[{}] Input shape error: Flatten layer "
            "requires an input with shape greater than 2.".format(self.type))
        return True

    def get_output_shape_for(self, input_shape):
        """Shape of the output tensor

        Parameters
        ----------
        input_shape : tuple or list
            Shape of the input layer.

        Returns
        -------
        tuple
            Shape of the output tensor. ``(batch_size, total_num_input_units)``
        """
        return tuple([
            -1 if input_shape[0] is None else input_shape[0],
            np.prod(self.input_layer.get_shape()[1:].as_list())
        ])

    def get_output_for(self):
        """Ravel the input layer into a 1D vector

        Returns
        -------
        ``tf.Tensor``
            Output tensor of this layer.
        """
        with tf.name_scope(self.name) as scope:
            output = tf.reshape(
                self.input_layer,
                self.output_shape,
                name=scope
            )
        tf.add_to_collection(BerryKeys.LAYER_OUTPUTS, output)
        return output
