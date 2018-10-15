import numpy as np
import pandas as pd
from struct import Struct

BINARY_DATA_TYPES = {
    np.dtype('float32'): 'f',
    np.dtype('int32'): 'i',
    np.dtype('uint32'): 'I',
    np.dtype('int64'): 'q',
    np.dtype('uint64'): 'Q',
    np.dtype('int8'): 'b',
    np.dtype('uint8'): 'B',
}

def _binary_struct(dtypes):
    binary_data = '<' + ''.join([BINARY_DATA_TYPES[dtype] for dtype in dtypes])
    return Struct(binary_data)

def write_binary(stream, pc):
    s = _binary_struct(pc.dtypes)
    for row in pc.itertuples(index=False):
        stream.write(s.pack(*row))

def read_binary(stream, dtypes, count=None):
    s = _binary_struct(dtypes)
    size = s.size
    data = []
    while True:
        if count and len(data) + 1 == count: break
        row = stream.read(size)
        if len(row) < size: break
        data.append(s.unpack(row))
    return data
