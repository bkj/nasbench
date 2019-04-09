#!/bin/bash

# run.sh

wget https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord

# --
# Generate graphs

python nasbench/scripts/generate-all-graphs.py | gzip -c > data/all-graphs.gz

# !! `all-graphs.gz` has lots of duplicates, so this is ~4x slower than it could be
#    Takes ~ 10 minutes
zcat data/all-graphs.gz | \
    parallel --pipe -N 100 python nasbench/scripts/neighbors.py |\
    tqdm |\
    gzip -c > data/all-edges-nofilter.gz

zcat data/all-edges-nofilter.gz | sort -S50% | uniq | gzip -c > data/tmp.gz
mv data/tmp.gz data/all-edges-nofilter.gz