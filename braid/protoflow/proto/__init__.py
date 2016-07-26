from __future__ import absolute_import

try:
    from .berry_pb2 import *
except:
    raise ImportError(
        "Run `make` in 'protoflow/proto' folder to compile berry.proto")
