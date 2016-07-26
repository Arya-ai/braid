"""Convert prototxt to ``BerryModel``
"""
from __future__ import absolute_import
import os
import tensorflow as tf
from braid.berry import BerryModel
from braid.berry.layers import *
from braid.berry.optimizers import get_optimizer
from braid.berry.objectives import get_objective
from google.protobuf.text_format import Merge as ProtoMerge
from google.protobuf.internal.containers import RepeatedScalarFieldContainer
from .proto import NetParameter


class ProtoFlow(object):
    """Convert prototxt defined using Google Protobuf standard to a
    ``BerryModel``.

    Parameters
    ----------
    prototxt : string
        Could be path to a prototxt file or the model definition in protobuf
        format.

    input_layer : ``tf.Tensor``
        The input placeholder for the input model.

    num_classes : int
        Number of output classes/targets.

    Attributes
    ----------
    model : a ``berry.BerryModel`` class instance
        The BerryModel instance created from the layers defined in the
        prototxt.

    Examples
    --------
    >>> from protoflow import ProtoFlow
    >>> import tensorflow as tf
    >>> input_tens = tf.placeholder(tf.float32, shape=(None, 227, 227, 3))
    >>> proto_path = "protoflow/models/alexnet.pf"
    >>> num_classes = 50
    >>> parser = ProtoFlow(proto_path, input_tens, num_classes)
    >>> berry_model = parser.model
    """

    def __init__(self, prototxt, input_layer, num_classes):
        self._model = BerryModel()
        self._model.add(input_layer)
        self.num_classes = num_classes
        if os.path.exists(prototxt):
            with open(prototxt, 'r') as fp:
                txt = fp.read()
                fp.close()
        else:
            txt = prototxt
        pnet = ProtoMerge(txt, NetParameter())
        self.parse_proto(pnet)

    @property
    def model(self):
        return self._model

    def get_berry_layer(self, name):
        """Get the Berry layer class given the name of the class

        Parameters
        ----------
        name : string
            Name of a layer class in ``berry.layers``

        Returns
        -------
        ``berry.layers.Layer`` class
            The layer class with name ``name``
        """
        if name not in globals().keys():
            print "[ERROR] ProtoFlow: {} layer does not exist.".format(name)
        cls = globals()[name]
        return cls

    def dict_to_berry_layer(self, layer_info):
        """Create a ``berry.layers.Layer`` instance from dictionary
        containing layer arguments

        Parameters
        ----------
        layer_info : dict
            Dict containing information about the layer, such as name, type,
            and value of the arguments.

        Returns
        -------
        ``berry.layers.Layer`` instance
            A Berry layer configured according to the arguments in the dict
        """
        if layer_info.has_key('input'):
            layer_inputs = layer_info['input']
            if (isinstance(layer_inputs, list) and len(layer_inputs) == 1):
                layer_inputs = layer_inputs[0]
            if isinstance(layer_inputs, list):
                input_layer = []
                for l in layer_inputs:
                    input_layer.append(self._model.get(l))
            else:
                input_layer = self._model.get(layer_inputs)
        else:
            input_layer = self._model.last
        params = layer_info.get('params', {})
        if layer_info.has_key('name'):
            params['name'] = layer_info['name']
        layer_type = layer_info.pop('type')
        cls = self.get_berry_layer(layer_type)
        layer = cls(input_layer, **params)
        return layer

    def parse_proto_layer_fields(self, dlayer):
        """Parse the protobuf layer data structure and create a dictionary
        with the relevant layer information

        Parameters
        ----------
        dlayer : ``protoflow.proto.LayerParameter`` instance
            Protobuf data structure containing the layer information

        Returns
        -------
        dict
            Dict containing the relevant layer information
        """
        params = {}
        if not hasattr(dlayer, 'ListFields'):
            return dlayer
        for key, val in dlayer.ListFields():
            key = key.name
            if 'param' in key:
                key = 'params'
            if isinstance(val,
                          (int, str, float, RepeatedScalarFieldContainer)):
                if isinstance(val, RepeatedScalarFieldContainer):
                    val = list(val)
                params[key] = val
            else:
                params[key] = self.parse_proto_layer_fields(val)
        return params

    def parse_proto(self, pnet):
        """Parse the entire prototxt model

        Parameters
        ----------
        pnet : ``protoflow.proto.NetParameter`` instance
            Parse the prototxt file and load it using protobuf data structure.
        """
        num_layers = len(pnet.layer)
        for i, layer in enumerate(pnet.layer):
            layer_info = self.parse_proto_layer_fields(layer)
            if i == num_layers - 1:
                if layer_info['type'] == "Dense":
                    layer_info['params']['num_units'] = self.num_classes
            layer = self.dict_to_berry_layer(layer_info)
            self._model.add(layer)
