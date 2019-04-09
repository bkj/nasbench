#!/usr/bin/env python

"""
    export-graphs.py
"""


import pandas as pd

# --
# edges

edges = pd.read_csv('data/all-edges-nofilter.gz', header=None, sep=' ')
edges.columns = ('node1', 'node2')

# Drop neighbors that aren't in search space
sel = edges.node2.isin(edges.node1)
edges = edges[sel]

# Drop half the edges, for size
sel = edges.node1 <= edges.node2
edges = edges[sel]

edges = edges.sort_values(['node1', 'node2']).reset_index(drop=True)

edges.to_csv('data/edges.gz', index=None, header=None, sep='\t')

# --
# nodes

path = 'data/nasbench_only108.tfrecord'
api  = NASBench(path)

hashes = list(api.hash_iterator())
specs, results = list(zip(*[api.get_metrics_from_hash(h) for h in tqdm(hashes)]))

mean_final_training_time       = np.array([np.mean([rr['final_training_time'] for rr in r]) for r in tqdm(results)])
mean_final_validation_accuracy = np.array([np.mean([rr['final_validation_accuracy'] for rr in r]) for r in tqdm(results)])

nodes = pd.DataFrame({
    "hash"                           : hashes,
    "mean_final_validation_accuracy" : mean_final_validation_accuracy,
    "mean_final_training_time"       : mean_final_training_time,
})

nodes.to_csv('data/nodes.gz', index=None, sep='\t')

# --
# arches

graphs = pd.read_json('data/all-graphs.gz', lines=True)
graphs = graphs.sort_values('hash').reset_index(drop=True)
graphs = graphs[['hash', 'labeling', 'adj']]
graphs[['hash', 'adj', 'node_features']].to_csv('data/arches.gz', index=None, sep='\t')