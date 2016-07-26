import numpy as np
import tensorflow as tf
import argparse
import os
import time
from google.protobuf.text_format import Merge


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '../models/')


def one_hot(y, num_classes):
    Y = np.zeros((num_classes,), dtype=np.uint8)
    Y[y] = 1
    return Y


def test_proto(prototxt):
    import sys
    sys.path.append('../proto')
    from berry_pb2 import NetParameter
    with open(prototxt, 'r') as f:
        txt = f.read()
    net = Merge(txt, NetParameter())
    print net.ListFields()


def test_parser(prototxt, t):
    from protoflow import ProtoFlow
    from berry.layers import print_layers_summary
    from keras.datasets import mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[:, np.newaxis, ...]
    X_test = X_test[:, np.newaxis, ...]
    y_train = np.asarray([one_hot(y, 10) for y in list(y_train)])
    y_test = np.asarray([one_hot(y, 10) for y in list(y_test)])
    print X_train.shape, y_train.shape

    with tf.device('/gpu:2'):
        parser = ProtoFlow(prototxt, t, 100)
        model = parser.model
        print_layers_summary(model.layers)
        print[v.name for v in tf.trainable_variables()]


def arguments(bool_help=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('-proto', dest='func_proto', action='store_const',
                        const=test_proto)
    parser.add_argument('-p', dest='func_parser', action='store_const',
                        const=test_parser)
    parser.add_argument('-path', dest='prototxt')
    parser.add_argument('-vgg', dest='vgg', action='store_true')
    parser.add_argument('-alex', dest='alex', action='store_true')
    args = parser.parse_args()
    if bool_help:
        parser.print_help()
    return args

if __name__ == '__main__':
    args = arguments()
    prototxt = os.path.join(DATA_DIR, 'lenet.pt')
    t = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    if args.alex:
        prototxt = os.path.join(DATA_DIR, 'alexnet.pt')
        t = tf.placeholder(tf.float32, shape=(None, 227, 227, 3))
    elif args.vgg:
        prototxt = os.path.join(DATA_DIR, 'vgg.pt')
        t = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    elif args.prototxt != '':
        prototxt = args.prototxt
        t = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    try:
        flag = False
        for name, func in vars(args).items():
            if func is None or isinstance(func, (bool, str, unicode)):
                continue
            print '##############################'
            print name
            print '##############################'
            func(prototxt, t)
            print '##############################'
            flag = True
        if not flag:
            arguments(bool_help=True)

    except KeyboardInterrupt:
        print "Interrupting..."
