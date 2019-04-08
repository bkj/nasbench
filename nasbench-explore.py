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

import numpy as np
import pandas as pd
from tqdm import trange, tqdm

import sklearn
import sklearn.compose
import sklearn.impute
import sklearn.feature_selection
from sklearn.svm import LinearSVR

import nasbench
from nasbench.api import NASBench

INPUT      = 'input'
OUTPUT     = 'output'
CONV1X1    = 'conv1x1-bn-relu'
CONV3X3    = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

# --
# Helpers

cummin = np.minimum.accumulate
cummax = np.maximum.accumulate

# --
# IO

path = 'data/nasbench_only108.tfrecord'
api  = NASBench(path)

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
# Clean

class Prepper:
    def _make_edges(self, num_nodes):
        edges = []
        for idx in range(num_nodes ** 2):
            row = idx // num_nodes
            col = idx %  num_nodes
            if row < col:
                edges.append(idx)
                
        return np.array(edges)
    
    def featurize_one(self, s, edges):
        edge_features = list(np.hstack(s['module_adjacency'])[edges].astype(np.float64))
        node_features = s['module_operations']
        return edge_features + node_features
    
    def featurize_iter(self, specs, results, num_nodes=7):
        edges = self._make_edges(num_nodes=num_nodes)
        for s, r in tqdm(zip(specs, results), total=len(specs)):
            adj_num_nodes = s['module_adjacency'].shape[0]
            if adj_num_nodes == num_nodes:
                X    = self.featurize_one(s, edges)
                y    = np.mean([rr['final_validation_accuracy'] for rr in r])
                z    = np.mean([rr['final_test_accuracy'] for rr in r])
                cost = np.mean([rr['final_training_time'] for rr in r])
                yield X, y, z, cost
    
    def featurize_all(self, specs, results, num_nodes=7):
        gen = self.featurize_iter(specs, results, num_nodes=num_nodes)
        X, y, z, cost = list(zip(*gen))
        return X, np.array(y), np.array(z), np.array(cost)


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


X_5, y_5, z_5, cost_5 = Prepper().featurize_all(specs, results, num_nodes=5)
X_6, y_6, z_6, cost_6 = Prepper().featurize_all(specs, results, num_nodes=6)
X_7, y_7, z_7, cost_7 = Prepper().featurize_all(specs, results, num_nodes=7)

y_567    = np.hstack([y_5, y_6, y_7])
z_567    = np.hstack([z_5, z_6, z_7])
cost_567 = np.hstack([cost_5, cost_6, cost_7])

running_scores = {}
running_costs  = {}

running_scores[5], running_costs[5] = run_random(z_5, cost_5)
running_scores[6], running_costs[6] = run_random(z_6, cost_6)
running_scores[7], running_costs[7] = run_random(z_7, cost_7)
running_scores[0], running_costs[0] = run_random(z_567, cost_567)

# --
# Featurization

num_nodes = 7
X, y, z, cost = eval('X_%d, y_%d, z_%d, cost_%d' % (num_nodes, num_nodes, num_nodes, num_nodes))

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

train_budget = 1e6
test_budget  = 3e6

model_scores = []
for _ in trange(32):
    
    perm = np.random.permutation(Xf.shape[0])
    
    train_sel = perm[np.where(cost[perm].cumsum() <= train_budget)[0]]
    test_sel  = perm[np.where(cost[perm].cumsum() > train_budget)[0]]
    
    Xf_train, Xf_test     = Xf[train_sel], Xf[test_sel]
    y_train, _            = y[train_sel], y[test_sel]
    z_train, z_test       = z[train_sel], z[test_sel]
    cost_train, cost_test = cost[train_sel], cost[test_sel]
    
    model     = LinearSVR(C=1, max_iter=10000).fit(Xf_train, y_train)
    pred_test = model.predict(Xf_test)
    
    pred_rank = pred_test.argsort()[::-1]
    eval_sel  = pred_rank[cost_test[pred_rank].cumsum() < test_budget]
    
    train_best = y_train.max()
    eval_best  = z_test[eval_sel].max()
    
    best = max(train_best, eval_best)
    model_scores.append({
        "budget" : np.cumsum(np.hstack([cost_train, cost_test[eval_sel]])),
        "score"  : cummax(np.hstack([z_train, z_test[eval_sel]])),
    })
    # print(model_scores[-1]['score'][-1])

# --
# Plot

r_max = z_max
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


