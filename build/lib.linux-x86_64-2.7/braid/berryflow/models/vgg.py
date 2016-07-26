import os
import numpy as np
from math import sqrt


def build_berry_vgg16(input_layer, num_classes, model_params={}):
    from berry import layers, BerryModel

    nn = BerryModel()
    nn.add(input_layer)
    # Expected input_shape = (224, 224, 3)
    nn.add(layers.Convolution2D(
        nn.last, 64, 5, stride=2, pad='SAME', activation='relu',
        init='deepnet', b_val=0.0))
    nn.add(layers.Convolution2D(
        nn.last, 64, 3, pad='SAME', activation='relu',
        init='deepnet', b_val=0.0))
    nn.add(layers.MaxPooling2D(nn.last, 2, 2))

    nn.add(layers.Convolution2D(
        nn.last, 128, 3, pad='SAME', activation='relu',
        init='deepnet', b_val=0.0))
    nn.add(layers.Convolution2D(
        nn.last, 128, 3, pad='SAME', activation='relu',
        init='deepnet', b_val=0.0))
    nn.add(layers.MaxPooling2D(nn.last, 2, 2))

    nn.add(layers.Convolution2D(
        nn.last, 256, 3, pad='SAME', activation='relu',
        init='deepnet', b_val=0.0))
    nn.add(layers.Convolution2D(
        nn.last, 256, 3, pad='SAME', activation='relu',
        init='deepnet', b_val=0.0))
    nn.add(layers.Convolution2D(
        nn.last, 256, 3, pad='SAME', activation='relu',
        init='deepnet', b_val=0.0))
    nn.add(layers.MaxPooling2D(nn.last, 2, 2))

    nn.add(layers.Convolution2D(
        nn.last, 512, 3, pad='SAME', activation='relu',
        init='deepnet', b_val=0.0))
    nn.add(layers.Convolution2D(
        nn.last, 512, 3, pad='SAME', activation='relu',
        init='deepnet', b_val=0.0))
    nn.add(layers.Convolution2D(
        nn.last, 512, 3, pad='SAME', activation='relu',
        init='deepnet', b_val=0.0))
    nn.add(layers.MaxPooling2D(nn.last, 2, 2))

    nn.add(layers.Convolution2D(
        nn.last, 512, 3, pad='SAME', activation='relu',
        init='deepnet', b_val=0.0))
    nn.add(layers.Convolution2D(
        nn.last, 512, 3, pad='SAME', activation='relu',
        init='deepnet', b_val=0.0))
    nn.add(layers.Convolution2D(
        nn.last, 512, 3, pad='SAME', activation='relu',
        init='deepnet', b_val=0.0))
    nn.add(layers.MaxPooling2D(nn.last, 2, 2))

    nn.add(layers.Flatten(nn.last))
    nn.add(layers.Dense(nn.last, 4096, activation='relu',
                        W_stddev=1e-2, b_val=0.0))
    nn.add(layers.Dropout(nn.last, 0.5))
    nn.add(layers.Dense(nn.last, 4096, activation='relu',
                        W_stddev=1e-2, b_val=0.0))
    nn.add(layers.Dropout(nn.last, 0.5))
    nn.add(layers.Dense(
        nn.last, num_classes, activation='softmax',
        W_stddev=1e-3, b_val=0.0))

    return nn
