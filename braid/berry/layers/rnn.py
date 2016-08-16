"""Recurrent layers
"""
from __future__ import division
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
import warnings
from .base import Layer
from ..activations import get_activation
from .. import initializations as init
from ..utils import print_activations, get_convolve_shape, print_msg
from ..config import BerryKeys


__all__ = [
    "RNN",

]

class RNN(Layer):
    """RNN layer

    Parameters
    ----------
    incoming : :class:`Layer` or ``tf.Tensor``
        Parent layer, whose output is given as input to the current layer.

    num_units : int
        The number of output units.

    num_of_cells : int
        The number of cells/steps in the layer.

    cell_type : string, optional (default = "LSTM")
        Type of recurrent cell to be used.

    activation : string, optional (default = "linear")
        Nonlinearity to apply afer performing convolution. See
        :mod:`berry.activations`

    return_cell_out : bool, optional (default = False)
        If true output from all cells are returned as 3d tensor,
        otherwise just final output as 2d tensor.

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

    def __init__(self, incoming, num_units,num_of_cells,cell_type='LSTM',return_cell_out = False, use_peepholes=False, cell_clip=None, pad='VALID', activation=None,
                 init=None, W_stddev=1e-2, b_val=0.1, **kwargs):
        super(RNN, self).__init__(incoming, **kwargs)
        if activation:
            self.activation = get_activation(activation)
        else:
            self.activation = None
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
        self.cell_type = cell_type

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
        assert len(self.input_shape) == 3, (
            "{} Input shape error: 2D convolution "
            "requires input shape: (batch_size, "
            "num_of_cells, num_units)".format(self.type))
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
        """
        batch_size = input_shape[0]
        if self.return_cell_out:
            shape = tuple([batch_size,self.num_of_cells,self.num_units])
        else:
            shape = tuple([batch_size,self.num_units])
        return shape

    def get_output_for(self):
        """Perform the convolution operation, activation and return the output
        ``tf.Tensor``.

        Returns
        -------
        ``tf.Tensor``
            Output tensor of this layer.
        """
        states = []
        outputs = []
        lstm = rnn_cell.BasicLSTMCell(self.num_units, state_is_tuple=True)
        initial_state = state = lstm.zero_state(batch_size, tf.float32)
        with tf.name_scope(self.name) as scope:
            for _id in xrange(self.num_of_cells):
                if _id > 0:
                    scope.reuse_variables()
                output, state = lstm(self.input_layer, state)

                if self.activation is not None:
                    output = self.activation(output)

                outputs.append(output)
                states.append(state)

        final_state = state
        if self.return_cell_out:
            output = tf.reshape(tf.concat(1, outputs), [-1, size])
        else:
            output = outputs[-1]
        tf.add_to_collection(BerryKeys.LAYER_OUTPUTS, output)
        return output
