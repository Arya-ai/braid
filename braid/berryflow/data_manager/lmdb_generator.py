from __future__ import absolute_import
import numpy as np
import lmdb
import time
from ..proto import Datum
from ..misc import print_msg
from .preprocessing import (create_fixed_image_shape, one_hot)


class LMDBGenerator(object):
    """ Read image-label data from lmdb database and feed mini-batches to
    the model.

    Preprocessing:
        + Resize to a fixed size, `image_input_shape` with random fill
        + One hot `y` vectors - suitable for classification task
    """

    def __init__(self, transformer, model_input_shape, batch_size,
                 input_ph, target_ph, train_db_name, test_db_name=None):
        self.input_ph = input_ph.name
        self.target_ph = target_ph.name
        self.model_input_shape = model_input_shape
        self.batch_size = batch_size
        self._train_env = lmdb.open(train_db_name, readonly=True)
        self.test = False
        if test_db_name is not None:
            self._test_env = lmdb.open(test_db_name, readonly=True)
            self.test = True
        self._verify_transformer(transformer)
        self.transformer = transformer
        self.num_classes = self.transformer.num_classes
        self._stat_lmdb()

    def close(self):
        time.sleep(2)
        print_msg('Closing train_env and test_env.', obj=self, level='debug')
        try:
            self._train_env.close()
            if self.test:
                self._test_env.close()
        except:
            pass

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

    def _stat_lmdb(self):
        self.num_train_samples = int(self._train_env.stat()['entries'])
        self.num_test_samples = (int(self._test_env.stat()['entries'])
                                 if self.test else 0)
        # Print params
        print_msg('Dataset statistics:', level='info', obj=self)
        print '\tNum of train samples: %d' % (
            self.num_train_samples)
        if self.test and self.num_test_samples == 0:
            self.test = False
        else:
            print '\tNum of test samples: %d' % (
                self.num_test_samples)
        print '\tNum of classes:', self.num_classes
        return

    def _verify_batch_data(self, X, Y):
        outX = np.asarray(X)
        if outX.ndim == 3:
            outX = outX[..., np.newaxis]
        # outX = np.transpose(outX, axes=(0, 3, 1, 2))
        outY = np.asarray(Y)
        if not outX.shape[1:] == tuple(self.model_input_shape)[1:]:
            print_msg("generated mini-batch shape {} mis-matched with expected"
                      " model input shape {}".format(outX.shape,
                                                     self.model_input_shape),
                      obj=self)
        return outX, outY

    def _parse_data_from_string(self, value):
        datum = Datum()
        datum.ParseFromString(value)
        flat_x = np.fromstring(datum.data, dtype=np.uint8)
        x_ = flat_x.reshape(datum.channels, datum.height, datum.width)
        x = np.transpose(x_, axes=(1, 2, 0))
        y = datum.label
        return x, y

    def train_batches_from_lmdb(self):
        X = []
        Y = []
        while True:
            with self._train_env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    try:
                        x_, y_ = self._parse_data_from_string(value)
                        x, y = self.transformer.transform(x_, y_, 'train')
                        if len(X) == self.batch_size:
                            outX, outY = self._verify_batch_data(X, Y)
                            if not (outX.shape[0] == self.batch_size and
                                    len(outY) == self.batch_size):
                                print_msg("batch size mismatch error",
                                          obj=self)
                            yield {self.input_ph: outX, self.target_ph: outY}
                            X[:] = []
                            Y[:] = []
                        X.append(x)
                        Y.append(y)
                    except Exception as e:
                        print_msg("Error generating data!!\n{}".format(e),
                                  obj=self, level=Exception)

    def test_batches_from_lmdb(self):
        if not self.test:
            print_msg("test database not initialized;"
                      " provide test_db_name during class init",
                      obj=self, level=AttributeError)
        X = []
        Y = []
        while True:
            with self._train_env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    try:
                        x_, y_ = self._parse_data_from_string(value)
                        x, y = self.transformer.transform(x_, y_, 'test')
                        if len(X) == self.batch_size:
                            outX, outY = self._verify_batch_data(X, Y)
                            if not (outX.shape[0] == self.batch_size and
                                    len(outY) == self.batch_size):
                                print_msg("batch size mismatch error",
                                          obj=self)
                            yield {self.input_ph: outX, self.target_ph: outY}
                            X[:] = []
                            Y[:] = []
                        X.append(x)
                        Y.append(y)
                    except Exception as e:
                        print_msg("Error generating data!!\n{}".format(e),
                                  obj=self, level=Exception)
