from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import time
from math import floor, ceil
from ..misc import print_msg
from .preprocessing import (create_fixed_image_shape, one_hot)


class VanillaGenerator(object):
    """ Read image-label data from lmdb database and feed mini-batches to
    the model.

    Preprocessing:
        + Resize to a fixed size, `image_input_shape` with random fill
        + One hot `y` vectors - suitable for classification task
    """

    def __init__(self, num_classes, batch_size,
                 X_train=None, y_train=None, input_ph=None, target_ph=None,
                 X_test=None, transformer=None):
        self.num_classes = num_classes
        self.batch_size = batch_size
        if not isinstance(input_ph, (tf.Tensor, tf.Variable)):
            print_msg("Input placeholder 'input_ph' expected", obj=self)
        self.input_ph = input_ph.name
        if target_ph is None:
            print_msg("Target placeholder 'target_ph' expected", obj=self)
        self.target_ph = target_ph.name

        self.train = False
        if X_train is not None and y_train is not None:
            self.X_train = np.asarray(X_train)
            self.y_train = np.asarray(y_train)
            if X_train.shape[0] != y_train.shape[0]:
                print_msg("Unequal number of samples in 'X_train' "
                          "and 'y_train'", obj=self)
            self.num_train_samples = self.X_train.shape[0]
            self.train = True
        elif X_test is not None:
            self.X_test = np.asarray(X_test)
            self.num_test_samples = self.X_test.shape[0]
        else:
            print_msg("Either 'X_train' and 'y_train' should be provided or "
                      "'X_test' should be provided", obj=self)

        if transformer is not None:
            self._verify_transformer(transformer)
        self.transformer = transformer

        self._print_stats()

    def close(self):
        time.sleep(2)

    def _verify_transformer(self, transformer):
        if not hasattr(transformer, 'transform'):
            print_msg("Expected class with a `transform` method; "
                      "check `Transformer` class", obj=self)
        try:
            dummy_shape = [100, 100]
            if transformer.reshape:
                if len(transformer.reshape) > 2:
                    dummy_shape = dummy_shape + [transformer.reshape[2]]
                else:
                    dummy_shape = dummy_shape
            else:
                dummy_shape = dummy_shape + [3]
            dummy = np.zeros(tuple(dummy_shape), dtype=np.uint8)
            dum_x, dum_y = transformer.transform(dummy, 0, 'train')
            if not (isinstance(dum_x, np.ndarray) and dum_x.ndim in {2, 3}
                    and isinstance(dum_y, np.ndarray) and dum_y.ndim == 1):
                print_msg("`transform` should return"
                          " (x, y) 2 `numpy.ndarray`'s, "
                          "`x.ndim = 2 or 3` and `y.ndim = 1`", obj=self)
        except TypeError as e:
            print_msg("{}".format(e), obj=self, level=TypeError)
        except ValueError as v:
            print_msg("`ValueError` caught, "
                      "`transform` should return (x, y) 2 `numpy.ndarray`'s, "
                      "`x.ndim = 2 or 3` and `y.ndim = 1`",
                      obj=self, level=ValueError)

    def _print_stats(self):
        # Print params
        print_msg('Dataset statistics:', level='info', obj=self)
        if self.train:
            print_msg('\tNum of train samples: {}'.format(
                self.num_train_samples), obj=self, level='info')
        else:
            print_msg('\tNum of test samples: {}'.format(
                self.num_test_samples), obj=self, level='info')
        print_msg('\tNum of classes: {}'.format(self.num_classes),
                  obj=self, level='info')

    def _prepare_batch(self, data, labels=None, mode='test'):
        X = []
        Y = []
        for i in range(data.shape[0]):
            x_ = data[i, ...]
            y_ = 0
            if self.transformer is not None:
                if labels is not None:
                    y_ = self.labels[i]
                x, y = self.transformer.transform(x_, y_, mode)
            else:
                x = x_
                y = np.zeros((self.num_classes), np.uint16)
            X.append(x)
            Y.append(y)
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim == 3:
            X = X[..., np.newaxis]
        if X.shape[0] != data.shape[0]:
            print_msg("Failed shape assertion while creating mini-batch",
                      obj=self)
        if Y.shape[1] != self.num_classes:
            print_msg("Y shape expected (batch_size, num_classes), "
                      "but got {}".format(Y.shape), obj=self)
        if X.shape[0] != Y.shape[0]:
            print_msg("data {} and labels {} shape mis-match".format(
                X.shape, Y.shape), obj=self)
        # X = np.transpose(X, (0, 3, 1, 2))
        feeder = {self.input_ph: X, self.target_ph: Y}
        return feeder

    def _loop(self, X, Y, num_samples, mode):
        batch_size = self.batch_size
        seen_samples = 0
        num_batches = int(ceil(num_samples / batch_size))
        while True:
            for i in range(num_batches):
                X_ = X[i * batch_size:(i + 1) * batch_size, ...]
                Y_ = None if Y is None else Y_[
                    i * batch_size:(i + 1) * batch_size]
                feed_dict = self._prepare_batch(X_, Y_, mode=mode)
                yield feed_dict
                seen_samples += batch_size
            remaining = num_samples - seen_samples
            if remaining > 0:
                X_ = X[-remaining:, ...]
                Y_ = None if Y is None else Y_[-remaining:]
                feed_dict = self._prepare_batch(X_, Y_, mode=mode)
                yield feed_dict

    def batch_generator(self):
        if self.train:
            return self._loop(self.X_train, self.y_train,
                              self.num_train_samples, 'train')
        else:
            return self._loop(self.X_test, None,
                              self.num_test_samples, 'test')
