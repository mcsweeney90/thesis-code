#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert NetworkX schedule graphs from Chapter 2 to StochDAGs and save.
"""

import dill, pathlib
from itertools import product

import sys
sys.path.append("../")
from src import RV, StochDAG

ntasks = list(range(5, 11, 5))
for nt, nb, s in product(ntasks, [128, 1024], [1, 4]):
    print(nt)
    chol_load_path = '../../chapter2/chapter4/nb{}s{}/'.format(nb, s) # TODO: double check this.
    chol_save_path = 'nb{}s{}/'.format(nb, s)
    pathlib.Path(chol_save_path).mkdir(parents=True, exist_ok=True)
    
    with open('{}/{}.dill'.format(chol_load_path, nt), 'rb') as file:
        R = dill.load(file)
    
    # Convert graph to ScaDAG.
    A = R.__class__()
    A.add_nodes_from(R)
    A.add_edges_from(R.edges)
    
    for t in R.nodes:
        mu, var = R.nodes[t]['weight']
        A.nodes[t]['weight'] = RV(mu, var)
    
    for u, v in R.edges:
        try:
            mu, var = R[u][v]['weight']
            A[u][v]['weight'] = RV(mu, var)
        except TypeError:
            A[u][v]['weight'] = 0.0
    
    G = StochDAG(A)
    # Save for future use. (Commented out by default to prevent overwriting data used for thesis.)
    # with open('{}/{}.dill'.format(chol_save_path, nt), 'wb') as handle:
    #     dill.dump(G, handle)
            
