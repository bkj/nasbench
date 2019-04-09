#!/usr/bin/env python

"""
  generate-all-graphs.py
  
  python generate-all-graphs.py | gzip -c > all-graphs.gz
"""

import sys
import json
import itertools
import numpy as np
from tqdm import tqdm
from nasbench.lib import graph_util
from joblib import delayed, Parallel

max_vertices = 7
num_ops      = 3
max_edges    = 9

def make_graphs(vertices, bits):
    matrix = np.fromfunction(graph_util.gen_is_edge_fn(bits), (vertices, vertices), dtype=np.int8)
    
    if graph_util.num_edges(matrix) > max_edges:
        return []
    
    if not graph_util.is_full_dag(matrix):
        return []
    
    out = []
    for labeling in itertools.product(*[range(num_ops) for _ in range(vertices-2)]):
      labeling = [-1] + list(labeling) + [-2]
      
      out.append({
        "hash"     : graph_util.hash_module(matrix, labeling),
        "adj"      : matrix.tolist(), 
        "labeling" : labeling,
      })
    
    return out


adjs = []
for vertices in range(2, max_vertices+1):
  for bits in range(2 ** (vertices * (vertices-1) // 2)):
    adjs.append((vertices, bits))

adjs = [adjs[i] for i in np.random.permutation(len(adjs))]
jobs = [delayed(make_graphs)(*adj) for adj in adjs]
res  = Parallel(n_jobs=40, backend='multiprocessing', verbose=10)(jobs)

for r in res:
  for rr in r:
    print(json.dumps(rr))