import os
import numpy as np


def build_berry_alexnet(input_layer, num_classes, model_params={}):
    from berry import BerryModel
    from berry import layers

    nn = BerryModel()
    nn.add(input_layer)
    nn.add(layers.Convolution2D(
        nn.last, 96, 11, stride=4, pad='VALID', activation='relu',
        W_stddev=1e-2, b_val=0.1))
    nn.add(layers.MaxPooling2D(nn.last, 3, 2))
    nn.add(layers.Convolution2D(nn.last, 256, 5, pad='SAME', activation='relu',
                                W_stddev=1e-2, b_val=0.1))
    nn.add(layers.MaxPooling2D(nn.last, 3, 2))
    nn.add(layers.Convolution2D(nn.last, 384, 3, pad='SAME', activation='relu',
                                W_stddev=1e-2, b_val=0.1))
    nn.add(layers.Convolution2D(nn.last, 384, 3, pad='SAME', activation='relu',
                                W_stddev=1e-2, b_val=0.1))
    nn.add(layers.Convolution2D(nn.last, 256, 3, pad='SAME', activation='relu',
                                W_stddev=1e-2, b_val=0.1))
    nn.add(layers.MaxPooling2D(nn.last, 3, 2))
    nn.add(layers.Flatten(nn.last))
    nn.add(layers.Dense(
        nn.last, 4096, activation='relu', W_stddev=5e-3, b_val=0.1))
    nn.add(layers.Dropout(nn.last, .5))
    nn.add(layers.Dense(
        nn.last, 4096, activation='relu', W_stddev=5e-3, b_val=0.1))
    nn.add(layers.Dropout(nn.last, .5))
    nn.add(layers.Dense(
        nn.last, num_classes, activation='softmax', W_stddev=5e-3, b_val=0.1))
    return nn
