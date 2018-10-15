from . import binary_io

import numpy as np
import pandas as pd

PLY_DATA_TYPES = {
    np.dtype('float32'): 'float',
    np.dtype('float64'): 'double',
    np.dtype('int32'): 'int',
    np.dtype('uint32'): 'uint',
    np.dtype('int8'): 'char',
    np.dtype('uint8'): 'uchar',
}

def save_ply(fname, pc):
    with open(fname, 'wb') as f:
        return to_ply(f, pc)

def read_ply(fname):
    with open(fname, 'rb') as f:
        return from_ply(f)

def to_ply(f, pc):
    format = 'binary_little_endian'

    dtypes = pc.dtypes
    fields = pc.columns

    elements = "element vertex {size}\n".format(size=pc.shape[0])

    properties = ''.join([
        "property {dtype} {field}\n".format(dtype=PLY_DATA_TYPES[dtype], field=field)
        for dtype, field in zip(dtypes, fields)
    ])

    header = \
        "ply\nformat {format} 1.0\n".format(format=format) \
        + elements \
        + properties \
        + "end_header\n"
    
    f.write(header.encode('ascii'))

    binary_io.write_binary(f, pc)


def from_ply(f):
    if f.readline().decode().strip() != 'ply':
        raise IOError('file is not ply')
    format_header, format_name, format_version = f.readline().decode().strip().split()
    if format_header != 'format' and format_name not in ['binary_little_endian', 'ascii'] and format_version != "1.0":
        raise IOError('error parsing ply file: wrong format encoding')
    line = f.readline().decode()
    while line.strip().split()[0] in ['comment', 'obj_info']:
        line = f.readline().decode()

    element, vertex, count = line.strip().split()
    if element != 'element' and vertex != 'vertex':
        raise IOError('error parsing ply file: wrong element spec')
    count = int(count)
    
    properties = []
    dtypes = []
    
    while True:
        line = f.readline().decode()
        if line.strip() == 'end_header': break
        property, dtype, name = line.split()
        if property != 'property': 
            raise IOError('error parsing ply file: wrong property spec')
        dtype = [k for k,v in PLY_DATA_TYPES.items() if v == dtype][0]
        properties.append(name)
        dtypes.append(dtype)

    data = binary_io.read_binary(f, dtypes, count=count) if format_name == 'binary_little_endian' else ascii_io.read_ascii(f, properties)

    pc = pd.DataFrame(data, columns=properties)

    for p, t in zip(properties, dtypes):
        pc[p] = pc[p].astype(t)

    return pc
