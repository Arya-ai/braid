""" Helper class for constructing a neural network model using Berry
"""
from __future__ import division
import tensorflow as tf
from collections import OrderedDict
from .layers import Layer, get_all_aux_params
from .utils import print_msg

__all__ = [
    "BerryModel"
]


class BerryModel(object):
    """Helper class to create deep neural networks using Berry.

    :class:`berry.BerryModel` maintains an ordered dictionary of layers added
    to the graph and provides access to the output of the model. It also
    provides easy access to auxiliary variables and the values taken by them
    during 'train' and 'test' phase.

    Attributes
    ----------
    layers : ordered dict
        All the layers which have been added to the graph. "Key" is the name
        of the added object and the "value" is the object. Object can either
        be of :class:`berry.layers.Layer` class or ``tf.Tensor`` class.

    last : :class:`berry.layers.Layer` or ``tf.Tensor``
        The last layer added to the model.

    output : ``tf.Tensor``
        The output tensor of the last layer added to the model.

    train_aux : dict
        Contains the auxiliary inputs for the model. The "key" is the name of
        the auxiliary variables and the "value" is the value taken by the
        variable during the training phase.

    test_aux : dict
        Contains the auxiliary inputs for the model. The "key" is the name of
        the auxiliary variables and the "value" is the value taken by the
        variable during the testing phase.
    """

    def __init__(self):
        self._layers = OrderedDict()

    def add(self, layer_instance):
        """Add another layer to the model.

        Parameters
        ----------
        layer_instance : :class:`Layer` or ``tf.Tensor``
            Insert the layer to the ``dict`` of layers, ``self.layers``.
        """
        if not isinstance(layer_instance, (Layer, tf.Tensor)):
            print_msg("incorrect input type; expected 'berry.layers.Layer' "
                      "or 'tf.Tensor'", obj=self)
        self._layers[layer_instance.name] = layer_instance

    def get(self, layer_name):
        """Get an existing layer from the model.

        Parameters
        ----------
        layer_name : string
            Name of the layer.

        Returns
        -------
        :class:`Layer`
            The layer corresponding to the name, ``layer_name``.
        """
        if not self._layers.has_key(layer_name):
            print_msg("layer with '{}' name does not exist".format(
                layer_name), obj=self)
        return self._layers[layer_name]

    @property
    def layers(self):
        return self._layers.values()

    @property
    def last(self):
        if len(self._layers.keys()) == 0:
            print_msg("no layers added yet", obj=self)
        return self._layers[self._layers.keys()[-1]]

    @property
    def output(self):
        if isinstance(self.last, Layer):
            return self.last.output
        else:
            return self.last

    @property
    def train_aux(self):
        return get_all_aux_params(False)

    @property
    def test_aux(self):
        return get_all_aux_params(True)
