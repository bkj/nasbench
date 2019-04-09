#!/usr/bin/env python

"""
    oen-exchange.py
"""

import sys
import json
import numpy as np
from copy import copy, deepcopy

from nasbench.lib import graph_util

num_ops   = 3
max_edges = 9

for line in sys.stdin:
    arch = json.loads(line)
    
    hash_     = arch['hash']
    matrix    = np.vstack(arch['adj'])
    labeling  = deepcopy(arch['labeling'])
    
    # --
    # Operator exchanges
    
    for i in range(1, len(labeling) - 1):
        for op in range(num_ops):
            if labeling[i] == op:
                continue
            
            l = copy(labeling)
            l[i] = op
            
            print(hash_, graph_util.hash_module(matrix, l))
    
    # --
    # Edge exchanges
    
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            matrix[i,j] = 1 - matrix[i,j]
            print(hash_, graph_util.hash_module(matrix, labeling))
            matrix[i,j] = 1 - matrix[i,j]
    
    sys.stdout.flush()