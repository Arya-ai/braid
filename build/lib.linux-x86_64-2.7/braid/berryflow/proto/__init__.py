from __future__ import absolute_import

try:
    from .datum_pb2 import *
except:
    raise ImportError(
        "Run `make` in 'berryflow/proto' folder to compile datum.proto")
