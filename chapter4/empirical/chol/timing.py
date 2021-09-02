#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time how long it takes to generate emprirical longest path distributions using MC method. 
"""

import dill
import pandas as pd
from itertools import product
from timeit import default_timer as timer

import sys
sys.path.append("../../")
from src import StochDAG

runs = 10

data = []
ntasks = list(range(5, 51, 5))
for N, nb, s in product(ntasks, [128, 1024], [1, 4]): 
    chol_load_path = '../../chol_graphs/nb{}s{}'.format(nb, s)
    with open('{}/{}.dill'.format(chol_load_path, N), 'rb') as file:
        R = dill.load(file)
    G = StochDAG(R)
        
    for dist in ["normal", "gamma", "uniform"]:
        graph_data = {"n" : G.size, "N" : N, "nb" : nb, "s" : s, "DIST" : dist}
        
        for samps in [10, 100, 1000, 10000, 100000]:
            start = timer() 
            for _ in range(runs):
                _ = G.monte_carlo(samples=samps, dist=dist)
            elapsed = timer() - start
            graph_data[samps] = elapsed/runs
        data.append(graph_data)
        
# Save the dataframe.
df = pd.DataFrame(data)  
df.to_csv('timing.csv', encoding='utf-8', index=False)     

