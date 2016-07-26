"""Noise layers
"""
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
    "Dropout"
]


class Dropout(Layer):
    """Dropout layer

    Sets values to zero with probability p. See notes for disabling dropout
    during testing.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape

    p : float, optional (default = 0.5)
        The probability of setting a value to zero

    name : string, optional (default = None)
        Name of the layer. Should be specified for better readability (
        inherited from :class:`Layer`).

    Notes
    -----
    .. note::

        The dropout layer is a regularizer that randomly sets input values to
        zero; see [1]_, [2]_ for why this might improve generalization.

        For ease of use, see :func:`get_aux_inputs` to get the value of ``p``
        during training and testing.

    References
    ----------
    .. [1] Hinton, G., Srivastava, N., Krizhevsky, A., Sutskever, I.,
           Salakhutdinov, R. R. (2012):
           Improving neural networks by preventing co-adaptation of feature
           detectors. arXiv preprint arXiv:1207.0580.

    .. [2] Srivastava Nitish, Hinton, G., Krizhevsky, A., Sutskever,
           I., & Salakhutdinov, R. R. (2014):
           Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
           Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
    """

    def __init__(self, incoming, p, **kwargs):
        super(Dropout, self).__init__(incoming, **kwargs)
        self.p = p
        with tf.name_scope(self.name):
            self.keep_prob = tf.placeholder(tf.float32, name='p')
            self.keep_prob.train_val = self.p
            self.keep_prob.test_val = 1.0
        tf.add_to_collection(BerryKeys.AUX_INPUTS, self.keep_prob)

    def validate_input_layer(self, incoming):
        """Validate the input layer shape

        Returns ``True`` if the input layer is 2D else, raise an
        :py:exc:`exceptions.AssertError`.
        """
        assert len(self.input_shape) == 2, (
            "[{}] Input shape error: expected "
            "(batch_size, num_units)".format(self.type))
        return True

    def get_fan_out(self):
        """Output receptive field

        Returns
        -------
        int
            ``fan_in * p``
        """
        return int(self.fan_in * self.p)

    def get_aux_inputs(self):
        """This function returns the auxiliary inputs required for the
        layer.

        Returns
        -------
        list of tuples
            [(`tf.placeholder.name`, (train_phase_value, test_phase_value)),]

        Examples
        --------
        >>> l = Dropout(l_in, p=0.5)
        >>> print l.get_aux_inputs()
        [(u'drop_1/p:0', (0.5, 1.0))]
        """
        return [(self.keep_prob.name, (self.p, 1.0))]

    def get_output_shape_for(self, input_shape):
        """Shape of the output tensor

        Parameters
        ----------
        input_shape : tuple or list
            Shape of the input layer.

        Returns
        -------
        tuple
            Same as the ``input_shape``
        """
        return input_shape

    def get_output_for(self):
        """Perform the dropout operation and returns the output ``tf.Tensor``

        Returns
        -------
        ``tf.Tensor``
            Output tensor of this layer.
        """
        with tf.name_scope(self.name) as scope:
            output = tf.nn.dropout(
                self.input_layer,
                self.keep_prob,
                name=scope
            )
        tf.add_to_collection(BerryKeys.LAYER_OUTPUTS, output)
        return output
