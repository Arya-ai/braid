import os
import numpy as np
from keras.models import Graph
from keras.layers.core import (Dense, Dropout, Flatten)
from keras.layers.convolutional import (
    Convolution2D, MaxPooling2D)
from keras.optimizers import SGD


def build_berry_lenet(input_layer, num_classes, model_params={}):
    from berry import layers, BerryModel

    nn = BerryModel()
    nn.add(input_layer)
    nn.add(layers.Convolution2D(nn.last, 20, 5, pad='VALID',
                                activation='sigmoid', W_stddev=1e-1))
    nn.add(layers.MaxPooling2D(nn.last, 2, 2))
    nn.add(layers.Convolution2D(nn.last, 50, 5, pad='SAME', W_stddev=1e-1,
                                activation='sigmoid'))
    nn.add(layers.MaxPooling2D(nn.last, 2, 2))
    nn.add(layers.Flatten(nn.last))
    nn.add(layers.Dense(nn.last, 500, activation='relu', W_stddev=5e-3))
    nn.add(layers.Dense(nn.last, num_classes, activation='softmax',
                        W_stddev=5e-3))
    return nn
