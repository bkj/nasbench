#!/usr/bin/env python

"""
    simple-nasbench.py
    
    - Reproduce random search results from NASBENCH paper
        - https://arxiv.org/pdf/1902.09635.pdf
    
    - Simple NAS algorithm, which appears to strongly outperform algorithms from paper
        - Train LinearSVR on small random sample of architectures
        - Rank remaining architectures
        - Train in order
"""

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVR

from nasbench.api import NASBench

from rsub import *
from matplotlib import pyplot as plt

# --
# Helpers

cummin = np.minimum.accumulate
cummax = np.maximum.accumulate
cumsum = np.cumsum

def cumargmax(x):
    z = np.arange(x.shape[0], dtype=np.float)
    z[x != cummax(x)] = np.nan
    z = pd.Series(z).fillna(method='ffill')
    return z.values.astype(int)


def make_edge_sel(num_nodes=7):
    edges = []
    for idx in range(num_nodes ** 2):
        row = idx // num_nodes
        col = idx %  num_nodes
        if row < col:
            edges.append(idx)
            
    return np.array(edges)


def sample_one_column(x):
    i = np.arange(x.shape[0])
    j = np.random.choice(x.shape[1], x.shape[0], replace=True)
    return x[(i, j)]



max_nodes = 7
edge_sel  = make_edge_sel(num_nodes=max_nodes)

# --
# IO

path = 'data/nasbench_only108.tfrecord'
api  = NASBench(path)

# --
# ETL

hashes = np.array(list(api.hash_iterator()))

feats     = [None] * len(hashes)
results   = [None] * len(hashes)
test_acc  = [None] * len(hashes)
valid_acc = [None] * len(hashes)
cost      = [None] * len(hashes)
for i, h in tqdm(enumerate(hashes), total=len(hashes)):
    spec, result = api.get_metrics_from_hash(h)
    
    # --
    # Clean + featurize architecture
    
    num_nodes = spec['module_adjacency'].shape[0]
    padding   = max_nodes - num_nodes
    if padding > 0:
        spec['module_adjacency'] = np.pad(spec['module_adjacency'], 
            ((0,padding), (0,padding)), mode='constant')
        spec['module_operations'] += (['__noop__'] * padding)
        
    edge_feat = list(np.hstack(spec['module_adjacency'])[edge_sel].astype(np.float64))
    node_feat = spec['module_operations']
    feats[i]  = edge_feat + node_feat
    
    # --
    # Store results
    
    result = result[108]
    
    valid_acc[i] = [r['final_validation_accuracy'] for r in result]
    test_acc[i]  = [r['final_test_accuracy'] for r in result]
    cost[i]      = [r['final_training_time'] for r in result]

valid_acc = np.vstack(valid_acc)
test_acc  = np.vstack(test_acc)
cost      = np.vstack(cost)

mean_valid_acc = valid_acc.mean(axis=-1)
mean_test_acc  = test_acc.mean(axis=-1)
mean_cost      = cost.mean(axis=-1)

# --
# Featurize 

num_edges       = np.arange(max_nodes).sum()
nominal_indices = np.arange(num_edges, num_edges + num_nodes)

featurizer = make_pipeline(
    ColumnTransformer(
        transformers=[
            ('nominal', make_pipeline(
                OneHotEncoder(handle_unknown='ignore'),
            ), nominal_indices)
        ],
        remainder='passthrough',
    )
)

Xf = featurizer.fit_transform(feats)
Xf = Xf.astype(np.float64) / Xf.shape[1]

# --
# Reproducing random baseline

def run_simple_random(valid_acc, test_acc, cost, 
    models_per_run=1e4, n_repeats=1000, valid_mode='mean'):
    
    n_models = len(valid_acc)
    
    models_per_run = min(int(models_per_run), n_models)
    
    cum_valid_acc = [None] * n_repeats
    cum_test_acc  = [None] * n_repeats
    cum_costs     = [None] * n_repeats
    for i in trange(n_repeats):
        sel = np.random.choice(n_models, models_per_run, replace=False)
        
        valid_acc_sel = valid_acc[sel]
        test_acc_sel  = test_acc[sel]
        cost_sel      = cost[sel]
        
        if valid_mode == 'mean':
            valid_acc_sel = valid_acc_sel.mean(axis=-1)
            cost_mult     = 3
        elif valid_mode == 'sample_one':
            valid_acc_sel = sample_one_column(valid_acc_sel)
            cost_mult     = 1
        else:
            raise Exception
        
        test_acc_sel = test_acc_sel.mean(axis=-1)
        cost_sel     = cost_mult * cost_sel.mean(axis=-1)
        
        cum_valid_acc[i] = cummax(valid_acc_sel)
        cum_test_acc[i]  = test_acc_sel[cumargmax(valid_acc_sel)]
        cum_costs[i]     = cumsum(cost_sel)
        
    return (
        np.stack(cum_valid_acc),
        np.stack(cum_test_acc),
        np.stack(cum_costs),
    )


randm_cum_valid_acc, randm_cum_test_acc, randm_cum_costs = run_simple_random(
    valid_acc=valid_acc,
    test_acc=test_acc,
    cost=cost,
    valid_mode='mean',
)

mean_randm_cum_valid_acc = randm_cum_valid_acc.mean(axis=0)
mean_randm_cum_test_acc  = randm_cum_test_acc.mean(axis=0)
mean_randm_cum_costs     = randm_cum_costs.mean(axis=0)

rand1_cum_valid_acc, rand1_cum_test_acc, rand1_cum_costs = run_simple_random(
    valid_acc=valid_acc,
    test_acc=test_acc,
    cost=cost,
    valid_mode='sample_one'
)

mean_rand1_cum_valid_acc = rand1_cum_valid_acc.mean(axis=0)
mean_rand1_cum_test_acc  = rand1_cum_test_acc.mean(axis=0)
mean_rand1_cum_costs     = rand1_cum_costs.mean(axis=0)

# This agrees roughly w/ Fig7 in the paper
_ = plt.plot(mean_randm_cum_costs, mean_test_acc.max() - mean_randm_cum_test_acc, c='red', label='randm')

# for i in range(256):
#     _ = plt.plot(randm_cum_costs[i], mean_test_acc.max() - randm_cum_test_acc[i], c='red', alpha=0.01)

_ = plt.plot(mean_rand1_cum_costs, mean_test_acc.max() - mean_rand1_cum_test_acc, c='orange', label='rand1')
# for i in range(256):
#     _ = plt.plot(rand1_cum_costs[i], mean_test_acc.max() - rand1_cum_test_acc[i], c='orange', alpha=0.01)

_ = plt.xscale('log')
_ = plt.yscale('log')
_ = plt.ylim(1e-3, 1e-1)
_ = plt.legend()
_ = plt.grid(which='both', alpha=0.5)
show_plot()

# --
# Model baseline

def run_svr(Xf, valid_acc, test_acc, cost, 
    train_samples=100, n_repeats=32, C=1, valid_mode='mean'):
    
    n_models = len(valid_acc)
    
    cum_valid_acc = [None] * n_repeats
    cum_test_acc  = [None] * n_repeats
    cum_costs     = [None] * n_repeats
    for i in trange(n_repeats):
        
        perm = np.random.permutation(Xf.shape[0])
        train_sel, test_sel = perm[:train_samples], perm[train_samples:]
        
        Xf_train, Xf_test               = Xf[train_sel], Xf[test_sel]
        valid_acc_train, valid_acc_test = valid_acc[train_sel], valid_acc[test_sel]
        test_acc_train, test_acc_test   = test_acc[train_sel], test_acc[test_sel]
        cost_train, cost_test           = cost[train_sel], cost[test_sel]
        
        # >>
        if valid_mode == 'mean':
            valid_acc_train = valid_acc_train.mean(axis=-1)
            valid_acc_test  = valid_acc_test.mean(axis=-1)
            cost_mult       = 3
        elif valid_mode == 'sample_one':
            valid_acc_train = sample_one_column(valid_acc_train)
            valid_acc_test  = sample_one_column(valid_acc_test)
            cost_mult       = 1
        # <<
        
        test_acc_train = test_acc_train.mean(axis=-1)
        test_acc_test  = test_acc_test.mean(axis=-1)
        
        cost_train = cost_train.mean(axis=-1)
        cost_test  = cost_test.mean(axis=-1)
        
        model = LinearSVR(C=C, max_iter=int(1e5)).fit(Xf_train, valid_acc_train)
        
        pred_test = model.predict(Xf_test)
        rank_test = np.argsort(-pred_test)
        
        valid_acc_sel = np.concatenate([valid_acc_train, valid_acc_test[rank_test]])
        test_acc_sel  = np.concatenate([test_acc_train, test_acc_test[rank_test]])
        cost_sel      = np.concatenate([cost_train, cost_test[rank_test]])
        
        cum_valid_acc[i] = cummax(valid_acc_sel)
        cum_test_acc[i]  = test_acc_sel[cumargmax(valid_acc_sel)]
        cum_costs[i]     = cost_mult * cumsum(cost_sel)
    
    return (
        np.stack(cum_valid_acc),
        np.stack(cum_test_acc),
        np.stack(cum_costs),
    )


svr1_cum_valid_acc, svr1_cum_test_acc, svr1_cum_costs = run_svr(
    Xf=Xf,
    valid_acc=valid_acc,
    test_acc=test_acc,
    cost=cost,
    valid_mode='sample_one',
    C=1
)

mean_svr1_cum_valid_acc = svr1_cum_valid_acc.mean(axis=0)
mean_svr1_cum_test_acc  = svr1_cum_test_acc.mean(axis=0)
mean_svr1_cum_costs     = svr1_cum_costs.mean(axis=0)

svrm_cum_valid_acc, svrm_cum_test_acc, svrm_cum_costs = run_svr(
    Xf=Xf,
    valid_acc=valid_acc,
    test_acc=test_acc,
    cost=cost,
    valid_mode='mean',
    C=1
)

mean_svrm_cum_valid_acc = svrm_cum_valid_acc.mean(axis=0)
mean_svrm_cum_test_acc  = svrm_cum_test_acc.mean(axis=0)
mean_svrm_cum_costs     = svrm_cum_costs.mean(axis=0)


# plot random (same as above)
_ = plt.plot(mean_rand_cum_costs, mean_test_acc.max() - mean_rand_cum_test_acc,
    c='red', label='rand')
# for i in range(rand_cum_test_acc.shape[0]):
#     _ = plt.plot(rand_cum_costs[i], mean_test_acc.max() - rand_cum_test_acc[i], c='red', alpha=0.01)

# plot svr
_ = plt.plot(mean_svr1_cum_costs, mean_test_acc.max() - mean_svr1_cum_test_acc,
    c='blue', label='svr1')
# for i in range(svr_cum_test_acc.shape[0]):
#     _ = plt.plot(svr_cum_costs[i], mean_test_acc.max() - svr_cum_test_acc[i], c='blue', alpha=0.1)

_ = plt.plot(mean_svrm_cum_costs, mean_test_acc.max() - mean_svrm_cum_test_acc,
    c='green', label='svrm')

_ = plt.xscale('log')
_ = plt.yscale('log')
_ = plt.ylim(5e-4, 1e-1)
_ = plt.legend()
_ = plt.grid(which='both', alpha=0.5)
show_plot()



