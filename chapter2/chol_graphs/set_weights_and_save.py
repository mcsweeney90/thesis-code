#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set DAG weights and save. 
"""

import dill, src, pathlib
from timeit import default_timer as timer

dag_top_path = '../../graphs/cholesky/topologies/'
timings_path = 'timings/'

# Load timings.
with open('{}/timings.dill'.format(timings_path), 'rb') as file:
    timings = dill.load(file)
    
# Load DAG topology, set weights and save weighted DAGs for future use.
for nt in range(5, 51, 5):
    for nb in [128, 1024]:
        dag_save_path = '/nb{}'.format(nb)
        pathlib.Path(dag_save_path).mkdir(parents=True, exist_ok=True)
        start = timer()
        with open('{}/{}.dill'.format(dag_top_path, nt), 'rb') as file:
            top = dill.load(file)
        G = src.DAG(top)
        G.set_cholesky_weights(timings, nb=nb)
        with open('{}/{}.dill'.format(dag_save_path, nt), 'wb') as handle:
            dill.dump(G, handle)
        elapsed = timer() - start
        print("nt = {}, nb = {}, time = {}".format(nt, nb, elapsed))