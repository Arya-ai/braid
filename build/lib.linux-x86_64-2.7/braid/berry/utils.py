from __future__ import print_function
import tensorflow as tf
import sys
import warnings


def get_convolve_shape(inp, kernel, stride, p):
    if p == 'SAME':
        shape = inp
    elif p == 'VALID':
        shape = inp - kernel
    elif isinstance(p, int):
        shape = inp + 2 * p - kernel

    shape = (shape // stride) + 1
    return shape


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


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
