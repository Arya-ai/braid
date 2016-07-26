from __future__ import absolute_import, print_function
import numpy as np
import sys
import lmdb
from ..proto import Datum
import warnings


def custom_formatwarning(msg, *a):
    # ignore everything except the message
    return str(msg) + '\n'

warnings.formatwarning = custom_formatwarning


def train_test_shuffle_split(X, Y, num_samples, test_ratio):
    # Perform train test split
    if isinstance(test_ratio, float) and test_ratio < 1.:
        num_test = int(round(num_samples * test_ratio))
    else:
        num_test = round(test_ratio)
    idx = np.arange(num_samples)
    rng = np.random.RandomState(seed=12345)
    rng.shuffle(idx)
    train_X = X[idx[num_test:]]
    train_Y = Y[idx[num_test:]]
    test_X = X[idx[:num_test]]
    test_Y = Y[idx[:num_test]]
    return (train_X, train_Y), (test_X, test_Y)


def parse_data_from_string(value):
    datum = Datum()
    datum.ParseFromString(value)
    flat_x = np.fromstring(datum.data, dtype=np.uint8)
    x = flat_x.reshape(datum.channels, datum.height, datum.width)
    y = datum.label
    return x, y


def load_lmdb_dataset(db_name):
    X = []
    Y = []
    env = lmdb.open(db_name, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            datum = Datum()
            datum.ParseFromString(value)
            flat_x = np.fromstring(datum.data, dtype=np.uint8)
            X.append(flat_x.reshape(datum.channels, datum.height, datum.width))
            Y.append(datum.label)
    env.close()
    assert len(X) == len(Y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y


def print_msg(txt, obj=None, level=AssertionError):
    msg = "[{}] ".format(level.upper()) if isinstance(level, str) else ""
    msg += "{}: ".format(obj.__class__.__name__) if obj is not None else ""
    msg += txt
    if 'error' in str(level).lower() or 'exception' in str(level).lower():
        raise level(msg)
    elif str(level).lower() == 'warn':
        warnings.warn(msg, RuntimeWarning)
    else:
        print(msg)
    sys.stdout.flush()
