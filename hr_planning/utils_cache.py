# -*- coding: utf-8 -*-
'''
Copyright (c) 2021, MIT Interactive Robotics Group, PI Julie A. Shah.
Authors: Shen Li, Nadia Figueroa, Ankit Shah, Julie A. Shah
All rights reserved.
'''

import numpy as np
from IPython import embed
import yaml


def read_from_memmap(np_var_name, path):
    print("Read from", path)
    return np.memmap(
        path, dtype=np_var_name.dtype, mode='r', shape=np_var_name.shape)


def save_to_memmap(np_var_name, path):
    print("Save to", path)
    fp = np.memmap(
        path, dtype=np_var_name.dtype, mode='w+', shape=np_var_name.shape)
    fp[:] = np_var_name[:]
    del fp


def copy_yaml(in_path, out_path):
    assert in_path != out_path
    tmp = None
    with open(in_path) as f:
        tmp = yaml.load(f, Loader=yaml.FullLoader)
    with open(out_path, 'w') as f:
        yaml.dump(tmp, f)
