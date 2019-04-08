#!/usr/bin/env python

"""
    nasbench-explore.py
    
    !! Using validation accuracy, instead of test
    !! Using average over 4 folds, instead of sampling
    
    !! Not exploiting permutation invariance -- should enrich SVR training data
    !! Could train NN to learn a representation that's invariant to permutations
"""

from rsub import *
from matplotlib import pyplot as plt

import json
import pickle
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
from collections import defaultdict

import sklearn
import sklearn.compose
import sklearn.impute
import sklearn.feature_selection
from sklearn.svm import LinearSVR, SVR

import nasbench
from nasbench.api import NASBench

INPUT      = 'input'
OUTPUT     = 'output'
CONV1X1    = 'conv1x1-bn-relu'
CONV3X3    = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

label_lookup = {
    -1 : INPUT,
    -2 : OUTPUT,
    0  : CONV3X3,
    1  : CONV1X1,
    2  : MAXPOOL3X3,
}

# --
# Helpers

cummin = np.minimum.accumulate
cummax = np.maximum.accumulate

# --
# IO

path = 'data/nasbench_only108.tfrecord'
api  = NASBench(path)

graphs = pd.read_json('data/all-graphs.gz', lines=True)
graphs = graphs.sort_values('hash').reset_index(drop=True)
graphs = graphs[['hash', 'labeling', 'adj']]

graphs['num_nodes'] = graphs.labeling.apply(len)

# --
# ETL

hashes = list(api.hash_iterator())

specs, results = list(zip(*[api.get_metrics_from_hash(h) for h in tqdm(hashes)]))

results = [r[108] for r in results]

y_all = np.array([np.mean([rr['final_validation_accuracy'] for rr in r]) for r in tqdm(results)])
y_max = y_all.max()

z_all = np.array([np.mean([rr['final_test_accuracy'] for rr in r]) for r in tqdm(results)])
z_max = z_all.max()

(y_all[z_all.argmax()] < y_all).sum() # best test model has 174th best valid accuracy
# test acc           mean across 3 runs           0.943175752957662
# test acc           maximum across 3 runs        0.9466145634651184
# validation acc     mean across 3 runs           0.9505542318026224
# validation acc     maximum across 3 runs        0.9518229365348816

# --
# Clean data

def make_edges(num_nodes):
    edges = []
    for idx in range(num_nodes ** 2):
        row = idx // num_nodes
        col = idx %  num_nodes
        if row < col:
            edges.append(idx)
            
    return np.array(edges)

class Prepper:
    def featurize_one(self, s, edges):
        edge_features = list(np.hstack(s['module_adjacency'])[edges].astype(np.float64))
        node_features = s['module_operations']
        return edge_features + node_features
    
    def featurize_iter(self, specs, results, hashes, num_nodes=7):
        edges = make_edges(num_nodes=num_nodes)
        for s, r, h in tqdm(zip(specs, results, hashes), total=len(specs)):
            adj_num_nodes = s['module_adjacency'].shape[0]
            if adj_num_nodes == num_nodes:
                X     = self.featurize_one(s, edges)
                y     = np.mean([rr['final_validation_accuracy'] for rr in r])
                z     = np.mean([rr['final_test_accuracy'] for rr in r])
                cost  = np.mean([rr['final_training_time'] for rr in r])
                yield X, y, z, cost, h
    
    def featurize_all(self, specs, results, hashes, num_nodes=7):
        gen = self.featurize_iter(specs, results, hashes, num_nodes=num_nodes)
        X, y, z, cost, hash_ = list(zip(*gen))
        return X, np.array(y), np.array(z), np.array(cost), np.array(hash_)

# Clean data
X_5, y_5, z_5, cost_5, hash_5 = Prepper().featurize_all(specs, results, hashes, num_nodes=5)
X_6, y_6, z_6, cost_6, hash_6 = Prepper().featurize_all(specs, results, hashes, num_nodes=6)
X_7, y_7, z_7, cost_7, hash_7 = Prepper().featurize_all(specs, results, hashes, num_nodes=7)

y_567    = np.hstack([y_5, y_6, y_7])
z_567    = np.hstack([z_5, z_6, z_7])
cost_567 = np.hstack([cost_5, cost_6, cost_7])
hash_567 = np.hstack([hash_5, hash_6, hash_7])

# --
# Clean graphs

# hash2feat = defaultdict(list)

# for num_nodes in [5, 6, 7]:
#     edges = make_edges(num_nodes=num_nodes)
#     sub   = graphs[graphs.num_nodes == num_nodes]
    
#     for idx, row in tqdm(sub.iterrows(), total=sub.shape[0]):
#         edge_features = list(np.hstack(row.adj)[edges].astype(np.float64))
#         node_features = [label_lookup.get(xx) for xx in row.labeling]
#         hash2feat[row.hash].append(edge_features + node_features)

# tmp = '\n'.join([json.dumps({"hash"  : k, "feats" : v}) for k,v in hash2feat.items()])
# _ = open('data/hash2feat.jl', 'w').write(tmp)

hash2feat = pd.read_json('data/hash2feat.jl', lines=True)
hash2feat = dict(zip(hash2feat.hash, hash2feat.feats))

# --
# Random search

def run_random(score, cost, models_per_run=1e4, n_repeats=500):
    num_models = score.shape[0]
    
    models_per_run = min(int(models_per_run), num_models)
    
    running_scores, running_costs = [], []
    for _ in trange(n_repeats):
        perm           = np.random.choice(num_models, models_per_run, replace=False)
        running_score  = cummax(score[perm])
        running_cost   = cost[perm].cumsum()
        
        running_scores.append(running_score)
        running_costs.append(running_cost)
        
    return np.stack(running_scores), np.stack(running_costs)

running_scores = {}
running_costs  = {}

running_scores[5], running_costs[5] = run_random(y_5, cost_5)
running_scores[6], running_costs[6] = run_random(y_6, cost_6)
running_scores[7], running_costs[7] = run_random(y_7, cost_7)
running_scores[0], running_costs[0] = run_random(y_567, cost_567)

# --
# Featurization

num_nodes = 6
X, y, z, cost, hash_ = eval('X_%d, y_%d, z_%d, cost_%d, hash_%d' % tuple([num_nodes] * 5))

num_edges       = np.arange(num_nodes).sum()
numeric_indices = np.arange(num_edges)
nominal_indices = np.arange(num_edges, num_edges + num_nodes)

featurizer = sklearn.pipeline.make_pipeline(
    sklearn.compose.ColumnTransformer(
        transformers=[
            # ('numeric', sklearn.pipeline.make_pipeline(
            #     sklearn.preprocessing.StandardScaler(),
            # ), numeric_indices),
            ('nominal', sklearn.pipeline.make_pipeline(
                sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore'),
            ), nominal_indices)
        ],
        remainder='passthrough',
    ),
    sklearn.feature_selection.VarianceThreshold(),
)

Xf = featurizer.fit_transform(X)
Xf = Xf.astype(np.float64)
Xf /= Xf.shape[1]

# --
# Simple search
#  - Train models for a certain amount of time
#  - Train a simple regression model
#  - Train suggested models for a certain amount of time

# !! These results aren't comparable to 

def flatten(x):
    out = [None] * sum(len(xx) for xx in x)
    idx = 0
    for xx in x:
        for xxx in xx:
            out[idx] = xxx
            idx += 1
    
    return out


augment_train = False
augment_test  = False

C = 1

train_budget = 1e5
test_budget  = 1e7

meds = []
for _ in range(10):
    model_scores = []
    for idx in range(32):
        
        perm = np.random.permutation(len(X))
        
        train_sel = perm[np.where(cost[perm].cumsum() <= train_budget)[0]]
        test_sel  = perm[np.where(cost[perm].cumsum() > train_budget)[0]]
        
        Xf_train, Xf_test     = Xf[train_sel], Xf[test_sel]
        hash_train, hash_test = hash_[train_sel], hash_[test_sel]
        y_train, y_test       = y[train_sel], y[test_sel]
        # z_train, z_test       = z[train_sel], z[test_sel]
        cost_train, cost_test = cost[train_sel], cost[test_sel]
        
        if not augment_train:
            model = LinearSVR(C=C, max_iter=10000).fit(Xf_train, y_train)
        else:
            X_train_aug  = sum([hash2feat[h] for h in hash_train], [])
            Xf_train_aug = featurizer.transform(X_train_aug).astype(np.float64)
            Xf_train_aug /= Xf_train_aug.shape[1]
            
            y_train_aug  = np.repeat(y_train, [len(hash2feat[h]) for h in hash_train])
            model = LinearSVR(C=C, max_iter=10000).fit(Xf_train_aug, y_train_aug)
        
        if not augment_test:
            pred_test = model.predict(Xf_test)
        else:
            X_test_aug  = flatten([hash2feat[h] for h in hash_test])
            Xf_test_aug = featurizer.transform(X_test_aug).astype(np.float64)
            Xf_test_aug /= Xf_test_aug.shape[1]
            
            key = np.repeat(hash_test, [len(hash2feat[h]) for h in hash_test])
            
            pred_test = model.predict(Xf_test_aug)
            pred_test = pd.Series(pred_test).groupby(key).mean()
            pred_test = pred_test.loc[hash_test].values
            
        
        pred_rank = pred_test.argsort()[::-1]
        eval_sel  = pred_rank[cost_test[pred_rank].cumsum() < test_budget]
        
        train_best = y_train.max()
        eval_best  = y_test[eval_sel].max()
        
        best = max(train_best, eval_best)
        model_scores.append({
            "budget" : np.cumsum(np.hstack([cost_train, cost_test[eval_sel]])),
            "score"  : cummax(np.hstack([y_train, y_test[eval_sel]])),
        })
        # print(model_scores[-1]['score'][-1])
        
    med = np.median([np.where(m['score'] == y_max)[0][0] for m in model_scores])
    print(med)
    meds.append(med)

print(meds)

# --
# Plot

r_max = y_max
for n, c in zip([0, 5, 6, 7], ['black', 'red', 'green', 'blue']):
    _ = plt.plot(running_costs[n].mean(axis=0), r_max - running_scores[n].mean(axis=0), c=c, label=n)
    # for running_score, running_cost in zip(running_scores[n], running_costs[n]):
    #     _ = plt.plot(running_cost, r_max - running_score, alpha=0.01, c=c)

for xx in model_scores:
    _ = plt.plot(xx['budget'], r_max - xx['score'], c='orange', alpha=0.25)


_ = plt.xscale('log')
_ = plt.yscale('log')
_ = plt.ylim(1e-4, 1e-1)
_ = plt.legend()
_ = plt.grid(which='both', alpha=0.5)
_ = plt.axvline(1e7, c='grey', alpha=0.5)
show_plot()


